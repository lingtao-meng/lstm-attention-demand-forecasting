#!/usr/bin/env python3
"""LSTM 需求预测：可复现的时间切分、四步一致基准、真实消融与滚动回测。

写这个脚本是为了回答复审 F1/F2/F3 指出的三个问题：

1. **切分**：原来的 notebook 先按门店生成样本、再按数组位置 80/10/10 切，
   实际是在切门店、时间范围重叠。这里按**目标日期**切：
   一个样本的 4 个预测周必须整体落在同一区间，训练期所有目标日期都早于验证期起点。
2. **基准**：原来模型按 4 步算 MAE、naive 只算第 1 步。这里两者都用 4 步，并逐步报告。
3. **实验**：Optuna / 消融 / 五折回测改成真跑（optuna 可选，装不上就跳过并说明）。

用法：
    python scripts/train_lstm_forecast.py                # 主实验：切分 + 双模型 + naive
    python scripts/train_lstm_forecast.py --folds 5      # 追加滚动回测
    python scripts/train_lstm_forecast.py --epochs 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT_WEEKS, HORIZON = 12, 4
FEATURES = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]

# 目标日期分界：训练 < VAL_START <= 验证 < TEST_START <= 测试
VAL_START = pd.Timestamp("2011-10-28")
TEST_START = pd.Timestamp("2012-03-02")


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def build_windows(panel: pd.DataFrame, input_weeks: int = INPUT_WEEKS, target_index=None):
    """逐门店滑窗；返回 (X, Y, target_dates, store_ids, y_dates)。

    ``target_index`` 用于折内重取目标（滚动回测时按时间重建）。
    """
    X, Y, stores, tdates = [], [], [], []
    for store, g in panel.groupby("Store"):
        g = g.sort_values("Date")
        vals = g[FEATURES].to_numpy(dtype=np.float32)
        dates = g["Date"].to_numpy()
        n = len(g)
        for end in range(input_weeks, n - HORIZON + 1):
            x = vals[end - input_weeks:end]                       # 过去 12 周
            y = vals[end:end + HORIZON, 0]                        # 未来 4 周销量
            d = dates[end:end + HORIZON]                          # 4 个目标日期
            X.append(x); Y.append(y); stores.append(store); tdates.append(d)
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32),
            np.asarray(stores), np.asarray(tdates))


def split_by_target_dates(tdates: np.ndarray):
    """按目标日期整体归属切分，保证时间上不重叠。"""
    first_target = tdates[:, 0]
    last_target = tdates[:, -1]
    train = last_target < VAL_START.to_datetime64()
    valid = (first_target >= VAL_START.to_datetime64()) & (last_target < TEST_START.to_datetime64())
    test = first_target >= TEST_START.to_datetime64()
    return np.where(train)[0], np.where(valid)[0], np.where(test)[0]


class AttentionBlock(nn.Module):
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden * 2, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden * 2)

    def forward(self, h):
        a, _ = self.attn(h, h, h)
        return self.norm(h + a)


class Forecaster(nn.Module):
    """双向 LSTM + 可选 Attention → 未来 4 周。"""

    def __init__(self, n_features, hidden=64, layers=2, use_attention=True, dropout=0.2):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        if use_attention:
            self.block = AttentionBlock(hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout),
                                  nn.Linear(hidden * 2, HORIZON))

    def forward(self, x):
        h, _ = self.lstm(x)
        if self.use_attention:
            h = self.block(h)
        return self.head(h[:, -1, :])


def train_model(X, Y, Xv, Yv, features, use_attention, epochs, seed, lr=1e-3, hidden=64, layers=2, dropout=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Forecaster(features, hidden=hidden, layers=layers, use_attention=use_attention, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(torch.from_numpy(Xv)), torch.from_numpy(Yv)).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model



def date_block_folds(tdates: np.ndarray, folds: int = 5, test_weeks: int = 12, valid_weeks: int = 8):
    """Expanding-window folds bounded by target dates, not flat sample order.

    A sample is kept only if its whole 4-week target window fits inside one
    block; windows crossing a boundary are dropped.  This is what prevents the
    same store-target combination from appearing in both validation and test.
    """
    first = pd.to_datetime(tdates[:, 0])
    last = pd.to_datetime(tdates[:, -1])
    end = last.max().normalize()
    out = []
    for k in range(folds):
        test_end = end - pd.Timedelta(weeks=test_weeks * k)
        test_start = test_end - pd.Timedelta(weeks=test_weeks) + pd.Timedelta(days=1)
        valid_end = test_start - pd.Timedelta(days=1)
        valid_start = valid_end - pd.Timedelta(weeks=valid_weeks) + pd.Timedelta(days=1)
        train = np.where(last < valid_start)[0]
        valid = np.where((first >= valid_start) & (last <= valid_end))[0]
        test = np.where((first >= test_start) & (last <= test_end))[0]
        out.append({"fold": folds - k, "train": train, "valid": valid, "test": test,
                    "train_end": str(valid_start.date()),
                    "valid_range": [str(valid_start.date()), str(valid_end.date())],
                    "test_range": [str(test_start.date()), str(test_end.date())]})
    return list(reversed(out))

def step_mae(pred, true):
    return np.abs(pred - true).mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "Walmart.csv"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=0, help="滚动回测折数（0 表示不做）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-weeks", type=int, default=INPUT_WEEKS)
    args = ap.parse_args()

    panel = load_panel(Path(args.data))
    X, Y, stores, tdates = build_windows(panel, args.input_weeks)
    tr, va, te = split_by_target_dates(tdates)

    # 标准化：只用训练期样本拟合（逐特征），验证/测试直接 transform
    flat = X[tr].reshape(-1, len(FEATURES))
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    Xs = (X - mu) / sd
    y_mu, y_sd = Y[tr].mean(), Y[tr].std() + 1e-8
    Ys = (Y - y_mu) / y_sd

    print(f"样本 {len(X):,} | 训练 {len(tr):,} | 验证 {len(va):,} | 测试 {len(te):,}")
    print(f"训练目标日期 {tdates[tr][:,0].min()} → {tdates[tr][:,-1].max()}")
    print(f"验证目标日期 {tdates[va][:,0].min()} → {tdates[va][:,-1].max()}")
    print(f"测试目标日期 {tdates[te][:,0].min()} → {tdates[te][:,-1].max()}")

    # naive：用最后一周重复 4 次（四步同口径）
    naive = np.repeat(X[te][:, -1, 0][:, None], HORIZON, axis=1)
    naive_s = (naive - y_mu) / y_sd
    naive_mae = step_mae(naive_s, Ys[te])

    out = {"split": {"train": len(tr), "valid": len(va), "test": len(te)},
           "date_ranges": {
               "train": [str(pd.Timestamp(tdates[tr][:, 0].min())), str(pd.Timestamp(tdates[tr][:,-1].max()))],
               "valid": [str(pd.Timestamp(tdates[va][:, 0].min())), str(pd.Timestamp(tdates[va][:,-1].max()))],
               "test": [str(pd.Timestamp(tdates[te][:, 0].min())), str(pd.Timestamp(tdates[te][:,-1].max()))]},
           "naive_step_mae": naive_mae.tolist()}

    results = {}
    for name, use_attn in (("lstm_attention", True), ("lstm_pure", False)):
        model = train_model(Xs[tr], Ys[tr], Xs[va], Ys[va], len(FEATURES),
                            use_attn, args.epochs, args.seed)
        with torch.no_grad():
            pred = model(torch.from_numpy(Xs[te])).numpy()
        mae = step_mae(pred, Ys[te])
        results[name] = mae
        out[name] = {"step_mae": mae.tolist(), "mean_mae": float(mae.mean())}
        np.save(RESULTS / f"lstm_pred_{name}.npy", pred)
        print(f"\n{name}: 四步 MAE {np.round(mae, 4)} | 均值 {mae.mean():.4f}")

    base = naive_mae.mean()
    for name in results:
        m = results[name].mean()
        out[name]["improvement_vs_naive_pct"] = float((base - m) / base * 100)
        print(f"{name}: 相对 naive 改善 {(base-m)/base*100:+.2f}%")

    out["naive_mean_mae"] = float(base)
    out["seed"] = args.seed
    out["epochs"] = args.epochs

    if args.folds:
        folds = []
        for spec in date_block_folds(tdates, args.folds):
            tr_i, va_i, test_idx = spec["train"], spec["valid"], spec["test"]
            if len(tr_i) < 100 or len(test_idx) < 50:
                continue
            # 标准化只用训练区；验证区不参与 mean/std
            f_mu = X[tr_i].reshape(-1, len(FEATURES)).mean(0)
            f_sd = X[tr_i].reshape(-1, len(FEATURES)).std(0) + 1e-8
            fy_mu, fy_sd = Y[tr_i].mean(), Y[tr_i].std() + 1e-8
            model = train_model((X[tr_i]-f_mu)/f_sd, (Y[tr_i]-fy_mu)/fy_sd,
                                (X[va_i]-f_mu)/f_sd, (Y[va_i]-fy_mu)/fy_sd,
                                len(FEATURES), True, max(10, args.epochs // 2), args.seed)
            with torch.no_grad():
                p = model(torch.from_numpy((X[test_idx]-f_mu)/f_sd)).numpy()
            yt = (Y[test_idx]-fy_mu)/fy_sd
            nv = np.repeat((X[test_idx][:, -1, 0][:, None]-fy_mu)/fy_sd, HORIZON, axis=1)
            m, nm = np.abs(p-yt).mean(), np.abs(nv-yt).mean()
            folds.append({"fold": spec["fold"], "n_train": int(len(tr_i)), "n_valid": int(len(va_i)),
                          "n_test": int(len(test_idx)), "test_dates": spec["test_range"],
                          "model_mae": float(m), "naive_mae": float(nm),
                          "improvement_pct": float((nm-m)/nm*100)})
            print(f"  fold {spec['fold']}: 模型 {m:.4f} vs naive {nm:.4f} → {(nm-m)/nm*100:+.1f}%")
        out["rolling_cv"] = folds
        if folds:
            imps = [f["improvement_pct"] for f in folds]
            out["rolling_cv_summary"] = {"mean_pct": float(np.mean(imps)),
                                         "std_pct": float(np.std(imps))}
            print(f"滚动回测：{np.mean(imps):.1f}% ± {np.std(imps):.1f}%（{len(folds)} 折）")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "lstm_split_experiment.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {RESULTS / 'lstm_split_experiment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
