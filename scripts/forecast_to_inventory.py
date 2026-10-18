#!/usr/bin/env python3
"""从预测误差到库存成本：把 MAE 的改善换算成钱。

需求预测项目最常见的断点：模型 MAE 降了 37.9%，然后呢？
计划岗真正关心的是这一步——**预测误差下降，能换来多少安全库存和持有成本的下降**。

做法（口径明确、可复现）：

1. 用 Walmart 数据算出 **naive 基线**（上周销量当本周预测）的真实误差分布；
2. 假设新模型的误差按同一比例下降（默认 ×0.621，即 MAE 降 37.9%）；
3. 安全库存 ``SS = z · σ_error · √L``（L = 提前期，周），持有成本 = SS × 持有成本率；
4. 输出：误差 → 安全库存 → 年持有成本 的换算，以及对服务水平的敏感性。

注意这是一个**情景推算**，不是企业实测：它回答的是「误差降 37.9% 意味着什么」，
而不是「我们公司省了多少钱」。

Usage
-----
    python scripts/forecast_to_inventory.py
    python scripts/forecast_to_inventory.py --improvement 0.379 --lead-time 4

Outputs go to ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_weekly_sales(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(["Store", "Date"])
    return df


def naive_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline: this week's forecast is last week's actual."""
    out = []
    for store, g in df.groupby("Store"):
        g = g.sort_values("Date")
        actual = g["Weekly_Sales"].to_numpy()
        pred = np.roll(actual, 1)
        err = actual[1:] - pred[1:]
        out.append(
            {
                "store": store,
                "n_weeks": len(err),
                "mae": float(np.abs(err).mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "mean_sales": float(actual.mean()),
            }
        )
    return pd.DataFrame(out)


def safety_stock(z: float, sigma_error: float, lead_time_weeks: float) -> float:
    """Safety stock in dollars of sales, for weekly demand with weekly lead time."""
    return z * sigma_error * np.sqrt(lead_time_weeks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "Walmart.csv"))
    ap.add_argument("--improvement", type=float, default=0.379,
                    help="relative MAE reduction of the new model (0.379 = 37.9%)")
    ap.add_argument("--lead-time", type=float, default=4.0, help="lead time in weeks")
    ap.add_argument("--holding-rate", type=float, default=0.25)
    ap.add_argument("--service-level", type=float, default=0.95)
    args = ap.parse_args()

    df = load_weekly_sales(Path(args.data))
    err = naive_errors(df)

    z = stats.norm.ppf(args.service_level)
    scale = 1.0 - args.improvement
    err["sigma_baseline"] = err["rmse"]
    err["sigma_model"] = err["rmse"] * scale
    err["ss_baseline"] = safety_stock(z, err["sigma_baseline"], args.lead_time)
    err["ss_model"] = safety_stock(z, err["sigma_model"], args.lead_time)
    err["holding_baseline"] = err["ss_baseline"] * args.holding_rate
    err["holding_model"] = err["ss_model"] * args.holding_rate

    tot = {
        "stores": len(err),
        "weeks": int(err["n_weeks"].sum()),
        "mae_baseline": err["mae"].sum(),
        "ss_baseline": err["ss_baseline"].sum(),
        "ss_model": err["ss_model"].sum(),
        "holding_baseline": err["holding_baseline"].sum(),
        "holding_model": err["holding_model"].sum(),
    }
    tot["ss_reduction"] = 1 - tot["ss_model"] / tot["ss_baseline"]
    tot["holding_saving"] = tot["holding_baseline"] - tot["holding_model"]

    print(f"stores={tot['stores']} weeks={tot['weeks']}")
    print(f"naive MAE (sum over stores, $/week): {tot['mae_baseline']:,.0f}")
    print(f"service level {args.service_level:.0%}, lead time {args.lead_time:.0f} weeks, "
          f"holding rate {args.holding_rate:.0%}")
    print(f"safety stock  baseline ${tot['ss_baseline']:,.0f} -> model ${tot['ss_model']:,.0f} "
          f"({-tot['ss_reduction']:.1%})")
    print(f"annual holding cost  ${tot['holding_baseline']:,.0f} -> ${tot['holding_model']:,.0f} "
          f"(saving ${tot['holding_saving']:,.0f})")

    # 服务水平敏感性
    rows = []
    for sl in (0.90, 0.95, 0.99):
        zz = stats.norm.ppf(sl)
        ss_b = safety_stock(zz, err["sigma_baseline"], args.lead_time).sum()
        ss_m = safety_stock(zz, err["sigma_model"], args.lead_time).sum()
        rows.append(
            {
                "service_level": sl,
                "ss_baseline": ss_b,
                "ss_model": ss_m,
                "holding_baseline": ss_b * args.holding_rate,
                "holding_model": ss_m * args.holding_rate,
                "annual_saving": (ss_b - ss_m) * args.holding_rate,
            }
        )
    sens = pd.DataFrame(rows)

    RESULTS.mkdir(exist_ok=True)
    err.to_csv(RESULTS / "forecast_to_inventory_per_store.csv", index=False)
    sens.to_csv(RESULTS / "forecast_to_inventory_sensitivity.csv", index=False)

    print("\nservice-level sensitivity:")
    print(sens.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    lines = [
        "# 预测误差 → 安全库存 → 持有成本",
        "",
        f"数据：Walmart 45 家门店 × {tot['weeks'] // max(tot['stores'], 1)} 周周度销售 "
        f"（naive 基线：上周销量）。",
        "",
        f"- 新模型相对 naive 的 MAE 改善：**{args.improvement:.1%}**（来自项目回测）",
        f"- 情景假设：模型误差按同一比例缩小，σ_model = {scale:.3f} × σ_naive",
        f"- 安全库存 `SS = z · σ_error · √L`，L = {args.lead_time:.0f} 周，"
        f"持有成本率 {args.holding_rate:.0%}/年",
        "",
        "| 指标 | naive 基线 | 新模型 | 变化 |",
        "|---|---|---|---|",
        f"| 单店周均绝对误差 MAE 合计 | ${tot['mae_baseline']:,.0f} | "
        f"${tot['mae_baseline'] * scale:,.0f} | {args.improvement:.0%} 更低 |",
        f"| 安全库存（{args.service_level:.0%} 服务水平，45 店合计） | "
        f"${tot['ss_baseline']:,.0f} | ${tot['ss_model']:,.0f} | **{tot['ss_reduction']:.1%} 更低** |",
        f"| 年持有成本 | ${tot['holding_baseline']:,.0f} | ${tot['holding_model']:,.0f} | "
        f"**省 ${tot['holding_saving']:,.0f}** |",
        "",
        "## 服务水平敏感性",
        "",
        sens.assign(service_level=(sens["service_level"] * 100).round(1))
        .to_string(index=False, float_format=lambda v: f"{v:,.0f}"),
        "",
        "> 口径说明：这是情景推算，不是企业实测。它回答「MAE 降 37.9% 意味着什么」，"
        "而不是「某公司省了多少钱」。安全库存对预测误差是线性关系，"
        "所以误差降幅会等比例传导到库存——这也是预测项目最容易被低估的价值。",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
