"""Train and check a six-window probabilistic M5 procurement backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_probabilistic import fit_quantile
from src.m5 import (
    add_baselines,
    add_features,
    feature_columns,
    load_sales,
    select_series,
    selection_history,
    weekly_panel,
)
from src.probabilistic import conformal_adjustment, conformal_quantile, pinball_loss

SEED = 42
SERIES = 100
TEST_WEEKS = 26
CAL_WEEKS = 16
FOLDS = 6
UNDERAGE = 0.30
OVERAGE = 0.25
ALPHAS = (0.1, 0.5, 0.9)
QUANTILES = ("q10", "q50", "q90")
POLICIES = ("p50_conformal", "naive_conformal", "naive_normal")
KEYS = ["fold", "id", "week_start"]


def fold_windows(sales: pd.DataFrame, calendar: pd.DataFrame):
    available = calendar.loc[calendar["d"].isin(sales.columns)]
    groups = available.groupby("wm_yr_wk")["date"]
    starts = pd.DatetimeIndex(groups.min()[groups.nunique().eq(7)].sort_values())
    if len(starts) <= FOLDS * TEST_WEEKS + CAL_WEEKS:
        raise ValueError("Insufficient complete M5 weeks for calibration and test")
    if not starts.to_series().diff().iloc[1:].eq(pd.Timedelta(weeks=1)).all():
        raise ValueError("M5 retail weeks are not consecutive")
    for fold in range(1, FOLDS + 1):
        end = len(starts) - (fold - 1) * TEST_WEEKS
        start = end - TEST_WEEKS
        yield fold, starts[start - CAL_WEEKS : start], starts[start:end]


def with_quantiles(frame: pd.DataFrame, features: list[str], models) -> pd.DataFrame:
    result = frame[["id", "week_start", "sales", "naive", "price"]].rename(columns={"sales": "actual"})
    for name, model in zip(QUANTILES, models, strict=True):
        result[name] = result["naive"] + model.predict(frame[features])
    return result


def score_fold(cal: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    for frame in (cal, test):
        if frame.empty or frame.duplicated(["id", "week_start"]).any():
            raise ValueError("Expected one nonempty prediction per item-store-week")
        if not np.isfinite(frame[["actual", "naive", "price", *QUANTILES]]).all().all():
            raise ValueError("Calibration and test inputs must be finite")
        if (frame["actual"] < 0).any() or (frame["price"] <= 0).any():
            raise ValueError("Expected nonnegative sales and positive prices")
    if cal["week_start"].max() >= test["week_start"].min():
        raise ValueError("Calibration must finish before the first test week")
    if not set(test["id"]).issubset(cal["id"]):
        raise ValueError("A test series has no pre-test calibration history")

    scores = np.maximum(cal["q10"] - cal["actual"], cal["actual"] - cal["q90"])
    adjustment = conformal_adjustment(scores.to_numpy(), 0.20)
    cr = UNDERAGE / (UNDERAGE + OVERAGE)
    score = test.copy()
    score["conformal_adjustment"] = adjustment
    score["lower80"] = score["q10"] - adjustment
    score["upper80"] = score["q90"] + adjustment
    naive_error = (cal["actual"] - cal["naive"]).groupby(cal["id"], observed=True)
    model_error = (cal["actual"] - cal["q50"]).groupby(cal["id"], observed=True)
    for name, forecast, errors in (
        ("p50_conformal", "q50", model_error),
        ("naive_conformal", "naive", naive_error),
        ("naive_normal", "naive", naive_error),
    ):
        if name == "naive_normal":
            # One observed residual has zero estimated spread for this comparator.
            offsets = errors.mean() + stats.norm.ppf(cr) * errors.std().fillna(0)
        else:
            offsets = errors.agg(lambda group: conformal_quantile(group.to_numpy(), cr))
        order = (score[forecast] + score["id"].map(offsets).astype(float)).clip(lower=0)
        score[f"order_{name}"] = order
        shortage = (score["actual"] - order).clip(lower=0)
        overage = (order - score["actual"]).clip(lower=0)
        score[f"shortage_{name}"] = shortage
        score[f"cost_{name}"] = score["price"] * (UNDERAGE * shortage + OVERAGE * overage)
    return score


def summarize(score: pd.DataFrame) -> dict:
    width = score["upper80"] - score["lower80"]
    actual = score["actual"]
    demand = float(actual.sum())
    result = {
        "test_rows": len(score),
        "coverage_80": float(actual.between(score["lower80"], score["upper80"]).mean()),
        "width_80": float(width.clip(lower=0).mean()),
        "empty_intervals": int(width.lt(0).sum()),
        "quantile_crossings": int(((score["q10"] > score["q50"]) | (score["q50"] > score["q90"])).sum()),
        **{
            f"pinball_{name}": pinball_loss(actual, score[name], alpha)
            for name, alpha in zip(QUANTILES, ALPHAS, strict=True)
        },
    }
    for name in POLICIES:
        result[f"cost_{name}"] = float(score[f"cost_{name}"].mean())
        result[f"stockout_share_{name}"] = float(score[f"shortage_{name}"].gt(0).mean())
        result[f"fill_rate_{name}"] = 1 - float(score[f"shortage_{name}"].sum()) / demand if demand else None
    naive = result["cost_naive_conformal"]
    result["cost_reduction"] = 1 - result["cost_p50_conformal"] / naive if naive else None
    return result


def read_evidence(file: Path) -> pd.DataFrame:
    return pd.read_csv(file, parse_dates=["week_start"], float_precision="round_trip")


def verify(outdir: Path) -> None:
    cal = read_evidence(outdir / "calibration.csv.gz")
    saved = read_evidence(outdir / "predictions.csv.gz")
    report = pd.read_csv(outdir / "folds.csv")
    info = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    folds = set(range(1, FOLDS + 1))
    if set(cal["fold"]) != folds or set(saved["fold"]) != folds or set(report["fold"]) != {0, *folds}:
        raise ValueError("Unexpected set of folds in the saved evidence")
    if sorted(window["fold"] for window in info["windows"]) != sorted(folds):
        raise ValueError("Expected one metadata window per fold")
    all_weeks = pd.DatetimeIndex(saved["week_start"].unique()).sort_values()
    if not all_weeks.equals(pd.date_range(all_weeks.min(), periods=FOLDS * TEST_WEEKS, freq="7D")):
        raise ValueError("Test windows must cover distinct consecutive retail weeks")
    if saved.duplicated(["id", "week_start"]).any():
        raise ValueError("An item-store-week is counted in more than one test fold")
    rescored = []
    rows = []
    for window in info["windows"]:
        fold = window["fold"]
        fold_cal, fold_test = cal[cal["fold"].eq(fold)], saved[saved["fold"].eq(fold)]
        for frame, label, weeks in ((fold_cal, "calibration", CAL_WEEKS), (fold_test, "test", TEST_WEEKS)):
            dates = pd.DatetimeIndex(frame["week_start"].unique()).sort_values()
            expected = pd.date_range(window[f"{label}_start"], periods=weeks, freq="7D")
            if not dates.equals(expected) or str(dates[-1].date()) != window[f"{label}_end"]:
                raise ValueError(f"Fold {fold}: unexpected {label} weeks in the evidence")
        expected_counts = (
            len(fold_cal) == window["calibration_rows"] <= window["raw_calibration_rows"] == SERIES * CAL_WEEKS
            and len(fold_test) == window["test_rows"] <= window["raw_test_rows"] == SERIES * TEST_WEEKS
            and fold_test["id"].nunique() == window["evaluated_series"] <= window["selected_series"] == SERIES
        )
        if not expected_counts:
            raise ValueError(f"Fold {fold}: saved row counts do not match the raw/eligible cohort")
        if not (
            window["selection_end"] < window["calibration_start"]
            and window["fit_end"] < window["calibration_start"] < window["test_start"]
        ):
            raise ValueError(f"Fold {fold}: invalid chronology")
        rebuilt = score_fold(fold_cal, fold_test[KEYS + ["actual", "naive", "price", *QUANTILES]])
        pd.testing.assert_frame_equal(
            fold_test.reset_index(drop=True), rebuilt.reset_index(drop=True), rtol=1e-10, atol=1e-10,
        )
        rescored.append(rebuilt)
        rows.append({"fold": fold, **summarize(rebuilt)})
    rows.append({"fold": 0, **summarize(pd.concat(rescored, ignore_index=True))})
    pd.testing.assert_frame_equal(
        report, pd.DataFrame(rows), check_dtype=False, rtol=1e-10, atol=1e-10,
    )
    print(f"Verified {len(cal):,} calibration rows and {len(saved):,} paired test rows across {FOLDS} folds.")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(data_dir: Path, outdir: Path) -> None:
    paths = {name: data_dir / name for name in ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")}
    sales = load_sales(paths["sales_train_validation.csv"])
    calendar = pd.read_csv(paths["calendar.csv"], parse_dates=["date"])
    prices = pd.read_csv(paths["sell_prices.csv"])
    outdir.mkdir(parents=True, exist_ok=True)
    rows, calibration_rows, scored_rows, windows = [], [], [], []
    for fold, cal_weeks, test_weeks in fold_windows(sales, calendar):
        days, history = selection_history(sales, calendar, gap_weeks=CAL_WEEKS, test_start=test_weeks[0])
        selected = select_series(sales, SERIES, SEED, selection_days=days)
        if len(selected) != SERIES:
            raise ValueError(f"Fold {fold}: got {len(selected)} eligible series, expected {SERIES}")
        # Reuse the same retail-week joins, features and quantile fit as the primary M5 exercise.
        panel = add_baselines(add_features(weekly_panel(selected, calendar, prices)))
        features = feature_columns(panel)
        raw_cal = panel.loc[panel["week_start"].isin(cal_weeks)]
        raw_test = panel.loc[panel["week_start"].isin(test_weeks)]
        if len(raw_cal) != SERIES * CAL_WEEKS or len(raw_test) != SERIES * TEST_WEEKS:
            raise ValueError(f"Fold {fold}: the raw calendar panel has missing or duplicate rows")
        ready = panel.dropna(subset=[*features, "sales"])
        train = ready.loc[ready["week_start"] < cal_weeks[0]]
        cal = ready.loc[ready["week_start"].isin(cal_weeks)]
        test = ready.loc[ready["week_start"].isin(test_weeks)]
        if train.empty or cal.empty or test.empty:
            raise ValueError(f"Fold {fold}: missing features left an empty split")
        print(f"Fold {fold}: fit/calibration/test rows {len(train):,}/{len(cal):,}/{len(test):,}", flush=True)
        models = [fit_quantile(train, features, alpha, SEED) for alpha in ALPHAS]
        cal_prediction = with_quantiles(cal, features, models).assign(fold=fold)
        test_prediction = with_quantiles(test, features, models).assign(fold=fold)
        columns = KEYS + ["actual", "naive", "price", *QUANTILES]
        cal_prediction = cal_prediction[columns]
        scored = score_fold(cal_prediction, test_prediction[columns])
        calibration_rows.append(cal_prediction)
        scored_rows.append(scored)
        rows.append({"fold": fold, **summarize(scored)})
        windows.append({
            "fold": fold,
            "selection_end": history["selection_end"],
            "fit_end": str(train["week_start"].max().date()),
            "calibration_start": str(cal_weeks[0].date()),
            "calibration_end": str(cal_weeks[-1].date()),
            "test_start": str(test_weeks[0].date()),
            "test_end": str(test_weeks[-1].date()),
            "fit_rows": len(train),
            "calibration_rows": len(cal),
            "raw_calibration_rows": len(raw_cal),
            "test_rows": len(test),
            "raw_test_rows": len(raw_test),
            "selected_series": len(selected),
            "evaluated_series": int(test["id"].nunique()),
            "conformal_adjustment": float(scored["conformal_adjustment"].iloc[0]),
        })
    pooled = pd.concat(scored_rows, ignore_index=True)
    rows.append({"fold": 0, **summarize(pooled)})
    pd.DataFrame(rows).to_csv(outdir / "folds.csv", index=False)
    compression = {"method": "gzip", "mtime": 0}
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        outdir / "calibration.csv.gz", index=False, compression=compression,
    )
    pooled.to_csv(outdir / "predictions.csv.gz", index=False, compression=compression)
    metadata = {
        "seed": SEED,
        "quantile_trees_per_model": 700,
        "series_per_fold": SERIES,
        "test_weeks": TEST_WEEKS,
        "calibration_weeks": CAL_WEEKS,
        "underage_price_ratio": UNDERAGE,
        "overage_price_ratio": OVERAGE,
        "data_sha256": {name: file_hash(path) for name, path in paths.items()},
        "environment": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "scipy", "lightgbm")
        }},
        "windows": windows,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    verify(outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/probabilistic_backtest")
    parser.add_argument("--verify", action="store_true", help="Check saved evidence without training models")
    args = parser.parse_args()
    if args.verify:
        verify(args.outdir)
    else:
        run(args.data_dir, args.outdir)


if __name__ == "__main__":
    main()
