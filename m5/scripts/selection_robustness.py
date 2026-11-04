"""Compare full-period and training-only M5 sample selection at fixed budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.m5 import add_baselines, add_features, feature_columns, load_sales, select_series, selection_history, time_split, weekly_panel  # noqa: E402
from src.probabilistic import conformal_adjustment, conformal_quantile, dm_test_hac  # noqa: E402
from run_analysis import train_model  # noqa: E402
from run_probabilistic import fit_quantile  # noqa: E402
from rolling_backtest import fit_model  # noqa: E402


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(selected, calendar, prices):
    frame = add_baselines(add_features(weekly_panel(selected, calendar, prices)))
    return frame, feature_columns(frame)


def predict_point(model, frame, features, direct=False):
    pred = model.predict(frame[features])
    return np.maximum(0, pred if direct else frame["naive"].to_numpy() + pred)


def score_point(frame, prediction):
    actual = frame["sales"].to_numpy()
    mae = {name: float(np.abs(actual - frame[name].to_numpy()).mean()) for name in ["naive", "ma4", "seasonal"]}
    mae["residual"] = float(np.abs(actual - prediction).mean())
    return {"mae": mae, "fva_naive": 1 - mae["residual"] / mae["naive"],
            "fva_best_baseline": 1 - mae["residual"] / min(mae[name] for name in ["naive", "ma4", "seasonal"])}


def base_arrays(test):
    return {"ids": test["id"].astype(str).to_numpy(dtype=str),
            "week_start": test["week_start"].dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
            **{c: test[c].to_numpy() for c in ["sales", "price", "naive", "ma4", "seasonal"]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results/robustness/selection")
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--series", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sales = load_sales(args.data_dir / "sales_train_validation.csv")
    calendar = pd.read_csv(args.data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(args.data_dir / "sell_prices.csv")
    config = {"seed": args.seed, "series": args.series, "holdout_weeks": args.holdout, "rolling_folds": args.folds,
              "selection_gap_weeks": 16, "calibration_weeks": 16, "point_validation_weeks": 8,
              "point_max_trees": 3000, "point_early_stopping": 100, "point_learning_rate": 0.03, "point_num_leaves": 63,
              "quantile_trees": 700, "quantile_learning_rate": 0.05, "quantile_num_leaves": 31,
              "underage": 0.30, "overage": 0.25, "dm_hac_lags": 4,
              "data_sha256": {name: fingerprint(args.data_dir / name) for name in ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]},
              "source_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in [ROOT / "src/m5.py", ROOT / "scripts/run_analysis.py", ROOT / "scripts/run_probabilistic.py", ROOT / "scripts/rolling_backtest.py"]},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__, "platform": platform.platform()},
              "scope": "Original point and quantile budgets; full-period fixed sample compared with sample selection before each independent 16-week calibration block. All folds refit except identical fold1/main reuse."}
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    full_selected = select_series(sales, args.series, args.seed)
    full_frame, features = prepare(full_selected, calendar, prices)
    week_table = full_frame[["week", "week_start"]].drop_duplicates().sort_values("week")
    weeks = week_table["week"].tolist()
    week_start_map = dict(zip(week_table["week"], week_table["week_start"]))
    results = {}
    for selection in ["full_period", "train_period"]:
        output = args.outdir / selection
        output.mkdir(exist_ok=True)
        if selection == "full_period":
            selected, frame = full_selected, full_frame
            metadata = {"selection_start": str(calendar[calendar["d"].isin(sales.columns)]["date"].min().date()),
                        "selection_end": str(calendar[calendar["d"].isin(sales.columns)]["date"].max().date()),
                        "uses_future_sales_for_selection": True}
        else:
            days, metadata = selection_history(sales, calendar, holdout=args.holdout, gap_weeks=16)
            metadata["uses_future_sales_for_selection"] = False
            selected = select_series(sales, args.series, args.seed, selection_days=days)
            frame, features = prepare(selected, calendar, prices)
        if len(selected) != args.series:
            raise ValueError("Sample did not contain the prespecified series count")
        selected[["id", "store_id", "item_id", "cat_id"]].to_csv(output / "selected_main.csv", index=False)
        overlap = len(set(selected["id"].astype(str)) & set(full_selected["id"].astype(str)))
        metadata["overlap_with_full_period_sample"] = overlap
        train, test, _ = time_split(frame, args.holdout)
        train = train.dropna(subset=features + ["sales"])
        test = test.dropna(subset=features).copy()
        print(f"start {selection} main point, train={len(train)} test={len(test)} overlap={overlap}", flush=True)
        main_model = train_model(train, features)
        main_prediction = predict_point(main_model, test, features)
        main_model.booster_.save_model(str(args.models_dir / f"{selection}_point.txt"))
        direct_model = train_model(train, features, direct=True)
        direct_prediction = predict_point(direct_model, test, features, direct=True)
        direct_model.booster_.save_model(str(args.models_dir / f"{selection}_direct.txt"))
        point = score_point(test, main_prediction)
        point["test_rows"] = len(test)
        point["raw_test_rows"] = int(frame[frame["week"].isin(test["week"].unique())].shape[0])
        point["mae"]["direct"] = float(np.abs(test["sales"].to_numpy() - direct_prediction).mean())
        point["best_iteration_residual"] = main_model.best_iteration_
        point["best_iteration_direct"] = direct_model.best_iteration_
        print(f"done {selection} point FVA={point['fva_naive']:+.2%}; start 3 quantile models x700 trees", flush=True)
        train_weeks = sorted(train["week"].unique())
        calibration_start = train_weeks[-16]
        proper = train[train["week"] < calibration_start]
        calibration = train[train["week"] >= calibration_start].copy()
        quantiles, cal_quantiles = {}, {}
        for alpha in [0.1, 0.5, 0.9]:
            model = fit_quantile(proper, features, alpha, args.seed)
            quantiles[alpha] = test["naive"].to_numpy() + model.predict(test[features])
            cal_quantiles[alpha] = calibration["naive"].to_numpy() + model.predict(calibration[features])
            model.booster_.save_model(str(args.models_dir / f"{selection}_q{alpha}.txt"))
        scores = np.maximum(cal_quantiles[0.1] - calibration["sales"].to_numpy(), calibration["sales"].to_numpy() - cal_quantiles[0.9])
        adjustment = conformal_adjustment(scores, 0.20)
        lower, upper = quantiles[0.1] - adjustment, quantiles[0.9] + adjustment
        calibration["model_residual"] = calibration["sales"].to_numpy() - cal_quantiles[0.5]
        calibration["base_residual"] = calibration["sales"] - calibration["naive"]
        cr = 0.30 / 0.55
        base_map = calibration.groupby("id", observed=True)["base_residual"].apply(lambda s: conformal_quantile(s.to_numpy(), cr))
        model_map = calibration.groupby("id", observed=True)["model_residual"].apply(lambda s: conformal_quantile(s.to_numpy(), cr))
        normal_map = calibration.groupby("id", observed=True)["base_residual"].apply(lambda s: s.mean() + stats.norm.ppf(cr) * s.std(ddof=1))
        quantities = {"naive_conformal": np.maximum(0, test["naive"].to_numpy() + test["id"].map(base_map).to_numpy()),
                      "p50_conformal": np.maximum(0, quantiles[0.5] + test["id"].map(model_map).to_numpy()),
                      "naive_normal": np.maximum(0, test["naive"].to_numpy() + test["id"].map(normal_map).to_numpy())}
        actual, price = test["sales"].to_numpy(), test["price"].to_numpy()
        costs = {name: float(np.mean(price * (0.30 * np.maximum(actual - q, 0) + 0.25 * np.maximum(q - actual, 0)))) for name, q in quantities.items()}
        probability = {"coverage_80": float(np.mean((actual >= lower) & (actual <= upper))),
                       "width_80": float(np.mean(upper - lower)), "conformal_adjustment": float(adjustment),
                       "quantile_crossings": int(np.sum((quantiles[0.1] > quantiles[0.5]) | (quantiles[0.5] > quantiles[0.9]))),
                       "costs": costs, "cost_reduction": 1 - costs["p50_conformal"] / costs["naive_conformal"],
                       "proper_rows": len(proper), "calibration_rows": len(calibration), "test_rows": len(test),
                       "calibration_start": str(calibration["week_start"].min().date()), "test_start": str(test["week_start"].min().date())}
        arrays = base_arrays(test)
        arrays.update(point=main_prediction, direct=direct_prediction, q10=quantiles[0.1], q50=quantiles[0.5], q90=quantiles[0.9], lower80=lower, upper80=upper)
        arrays.update({f"order_{name}": q for name, q in quantities.items()})
        arrays.update(cal_ids=calibration["id"].astype(str).to_numpy(dtype=str),
                      cal_week_start=calibration["week_start"].dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
                      cal_actual=calibration["sales"].to_numpy(), cal_naive=calibration["naive"].to_numpy(),
                      cal_q10=cal_quantiles[0.1], cal_q50=cal_quantiles[0.5], cal_q90=cal_quantiles[0.9])
        np.savez_compressed(output / "main_predictions.npz", **arrays)
        print(f"done {selection} probability cost improvement={probability['cost_reduction']:+.2%}", flush=True)
        folds = []
        for fold in range(1, args.folds + 1):
            end = len(weeks) - args.holdout * (fold - 1)
            fold_weeks = weeks[end - args.holdout:end]
            first_date = week_start_map[fold_weeks[0]]
            if selection == "train_period" and fold > 1:
                fold_days, fold_metadata = selection_history(sales, calendar, test_start=first_date, gap_weeks=16)
                fold_selected = select_series(sales, args.series, args.seed, selection_days=fold_days)
                fold_frame, fold_features = prepare(fold_selected, calendar, prices)
            else:
                fold_selected, fold_frame, fold_features, fold_metadata = selected, frame, features, metadata
            fold_selected[["id", "store_id", "item_id", "cat_id"]].to_csv(output / f"selected_fold_{fold}.csv", index=False)
            fold_train = fold_frame[fold_frame["week"] < fold_weeks[0]].dropna(subset=fold_features + ["sales"])
            raw_test = fold_frame[fold_frame["week"].isin(fold_weeks)]
            fold_test = raw_test.dropna(subset=fold_features).copy()
            if len(fold_selected) != args.series or len(raw_test) != args.series * args.holdout or len(fold_train) < 1000 or not len(fold_test):
                raise ValueError(f"Incomplete formal fold {selection}/{fold}: train={len(fold_train)} raw_test={len(raw_test)} evaluated_test={len(fold_test)}")
            print(f"start {selection} fold={fold} train={len(fold_train)} test={len(fold_test)}", flush=True)
            if fold == 1:
                if not fold_test.index.equals(test.index):
                    raise ValueError("Main/fold1 indices differ; cannot reuse training")
                prediction = main_prediction
                training_origin = "identical main residual training reused"
            else:
                model = fit_model(fold_train, fold_features)
                prediction = predict_point(model, fold_test, fold_features)
                model.booster_.save_model(str(args.models_dir / f"{selection}_fold_{fold}.txt"))
                training_origin = "refitted"
            stats_row = score_point(fold_test, prediction)
            weekly_loss = fold_test.assign(loss_model=np.abs(fold_test["sales"].to_numpy() - prediction),
                                           loss_base=np.abs(fold_test["sales"] - fold_test["naive"])).groupby("week")[["loss_model", "loss_base"]].mean()
            dm_stat, dm_p = dm_test_hac((weekly_loss["loss_model"] - weekly_loss["loss_base"]).to_numpy(), lags=4)
            row = {"fold": fold, "test_start": str(fold_test["week_start"].min().date()), "test_end": str(fold_test["week_start"].max().date()),
                   "train_rows": len(fold_train), "test_rows": len(fold_test), "raw_test_rows": len(raw_test),
                   "dropped_test_rows": len(raw_test) - len(fold_test),
                   "missing_feature_counts": {c: int(raw_test[c].isna().sum()) for c in fold_features if raw_test[c].isna().any()},
                   "evaluated_series": int(fold_test["id"].nunique()), "selected_series": len(fold_selected), "selection": fold_metadata,
                   "training_origin": training_origin, "dm_stat": dm_stat, "dm_p": dm_p, **stats_row}
            folds.append(row)
            fold_arrays = base_arrays(fold_test)
            fold_arrays["point"] = prediction
            np.savez_compressed(output / f"fold_{fold}_predictions.npz", **fold_arrays)
            (output / "folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"done {selection} fold={fold} FVA={row['fva_naive']:+.2%}", flush=True)
        fva = np.array([r["fva_naive"] for r in folds])
        result = {"selection": metadata, "point": point, "probability": probability, "folds": folds,
                  "rolling": {"folds": len(folds), "fva_mean": float(fva.mean()), "fva_std_ddof1": float(fva.std(ddof=1)),
                              "fva_min": float(fva.min()), "fva_max": float(fva.max()), "wins_vs_naive": int((fva > 0).sum())}}
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results[selection] = result
    results["elapsed_seconds"] = time.perf_counter() - started
    results["comparison_note"] = "Selection policies can produce different populations. Changes in absolute MAE/cost are not paired improvements on identical item-store samples. Within-policy FVA/cost comparisons use identical test records and independent calibration."
    (args.outdir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"point_fva": v["point"]["fva_naive"], "rolling": v["rolling"], "cost_reduction": v["probability"]["cost_reduction"]} for k, v in results.items() if isinstance(v, dict)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
