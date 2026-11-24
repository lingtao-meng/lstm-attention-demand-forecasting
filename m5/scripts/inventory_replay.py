"""Replay M5 replenishment policies using the six-fold forecast evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inventory import protection_offsets, simulate_inventory

LEADS = (1, 2)
LEVELS = (0.35, 0.50, 0.65, 0.80, 0.90, 0.95)
FORECASTS = ("naive", "q50")
FILL_THRESHOLDS = (0.90, 0.95, 0.97)
WARMUP = 4
FIXED_TAU = 0.65
BUFFER_FIT_WEEKS = 8
PROFILES = {
    "balanced": (0.02, 0.30),
    "holding_weighted": (0.04, 0.15),
    "shortage_weighted": (0.01, 0.45),
}
GROUP = ["lead", "forecast", "tau"]


def make_grid(frame: pd.DataFrame, weeks: int) -> tuple[pd.DatetimeIndex, list[str], dict[str, np.ndarray]]:
    dates = pd.DatetimeIndex(frame["week_start"].unique()).sort_values()
    ids = sorted(frame["id"].unique())
    if (
        len(dates) != weeks
        or not dates.equals(pd.date_range(dates.min(), periods=weeks, freq="7D"))
        or len(ids) != 100
        or len(frame) != len(dates) * len(ids)
    ):
        raise ValueError("M5 replay requires a complete regular week-by-series panel")
    matrices = {
        column: frame.pivot(index="week_start", columns="id", values=column).reindex(
            index=dates, columns=ids,
        ).to_numpy()
        for column in ("actual", "naive", "q50", "price")
    }
    if any(not np.isfinite(values).all() for values in matrices.values()) or (matrices["price"] <= 0).any():
        raise ValueError("Replay inputs must be finite with positive weekly prices")
    return dates, ids, matrices


def weekly_records(
    fold: int,
    dates: pd.DatetimeIndex,
    ids: list[str],
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    initial_history: np.ndarray,
    trace: list[pd.DataFrame] | None = None,
) -> list[pd.DataFrame]:
    result = []
    for lead in LEADS:
        protection = lead + 1
        initial = np.ceil(initial_history[-WARMUP:].mean(axis=0) * protection).astype(np.int64)
        for forecast in FORECASTS:
            for tau in LEVELS:
                offset = protection_offsets(calibration["actual"], calibration[forecast], protection, tau)
                projected = protection * np.maximum(test[forecast], 0)
                target = np.ceil(np.maximum(projected + offset, 0)).astype(np.int64)
                states = simulate_inventory(test["actual"], target, initial, lead)
                priced_stock = (test["price"] * states["closing_stock"]).sum(axis=1)
                priced_lost = (test["price"] * states["lost"]).sum(axis=1)
                rows = {
                    "fold": fold, "week_start": dates, "warmup": np.arange(len(dates)) < WARMUP,
                    "lead": lead, "forecast": forecast, "tau": tau,
                    "series": target.shape[1], "demand": test["actual"].sum(axis=1),
                    "stockout_series": (states["lost"] > 0).sum(axis=1),
                    **{name: values.sum(axis=1) for name, values in states.items()},
                    **{
                        f"cost_{name}": holding * priced_stock + shortage * priced_lost
                        for name, (holding, shortage) in PROFILES.items()
                    },
                }
                result.append(pd.DataFrame(rows))
                if trace is not None and tau == FIXED_TAU:
                    trace.append(pd.DataFrame({
                        "fold": fold, "lead": lead, "forecast": forecast, "tau": tau,
                        "week_start": dates.repeat(len(ids)), "id": np.tile(ids, len(dates)),
                        "warmup": np.repeat(rows["warmup"], len(ids)),
                        "demand": test["actual"].ravel(), "price": test["price"].ravel(),
                        "order_up_to": target.ravel(),
                        **{name: values.ravel() for name, values in states.items()},
                    }))
    return result


def summarize(weekly: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    measured = weekly.loc[~weekly["warmup"]]
    groups = measured.groupby(keys, sort=True)
    outcomes = groups[["series", "demand", "lost", "stockout_series", "closing_stock", "closing_pipeline",
                       "order", *(f"cost_{name}" for name in PROFILES)]].sum()
    result = pd.DataFrame({
        "scored_weeks": groups.size(), "series_weeks": outcomes["series"],
        "demand_units": outcomes["demand"], "lost_units": outcomes["lost"],
        "fill_rate": 1 - outcomes["lost"] / outcomes["demand"],
        "stockout_share": outcomes["stockout_series"] / outcomes["series"],
        "mean_closing_stock": outcomes["closing_stock"] / outcomes["series"],
        "mean_on_order": outcomes["closing_pipeline"] / outcomes["series"],
        "mean_order": outcomes["order"] / outcomes["series"],
        **{f"mean_cost_{name}": outcomes[f"cost_{name}"] / outcomes["series"] for name in PROFILES},
    })
    return result.reset_index()


def match_service(frontier: pd.DataFrame) -> pd.DataFrame:
    """Describe observed frontier points; no policy choice is validated on these same test weeks."""
    rows = []
    for (lead, forecast), group in frontier.groupby(["lead", "forecast"], sort=True):
        ordered = group.sort_values(["mean_closing_stock", "tau"])
        for level in FILL_THRESHOLDS:
            candidates = ordered.loc[ordered["fill_rate"].ge(level)]
            fields = ["tau", "fill_rate", "mean_closing_stock", "mean_cost_balanced"]
            best = {key: None for key in fields} if candidates.empty else candidates.iloc[0][fields].to_dict()
            rows.append({"lead": lead, "forecast": forecast, "minimum_fill_rate": level, **best})
    return pd.DataFrame(rows)


def evaluate_pretest_selection(tuning: pd.DataFrame, testing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fix tau from the later calibration block, then report only the corresponding test path.
    scored_tuning = summarize(tuning, ["fold", *GROUP])
    selection = []
    for fold, group in scored_tuning.groupby("fold", sort=True):
        selection.append(match_service(group).assign(fold=fold))
    selected = pd.concat(selection, ignore_index=True).rename(columns={
        "fill_rate": "tuning_fill_rate", "mean_closing_stock": "tuning_mean_closing_stock",
        "mean_cost_balanced": "tuning_mean_cost_balanced",
    })
    chosen = testing.merge(selected.dropna(subset=["tau"]), on=["fold", *GROUP], validate="m:m")
    test_keys = ["fold", "minimum_fill_rate", *GROUP]
    by_fold = selected.merge(summarize(chosen, test_keys), how="left", on=test_keys, validate="1:1")
    pair_keys = ["fold", "lead", "minimum_fill_rate"]
    paired = (
        selected.dropna(subset=["tau"]).groupby(pair_keys)["forecast"].nunique()
        .loc[lambda counts: counts.eq(len(FORECASTS))].reset_index()[pair_keys]
    )
    pooled_keys = ["lead", "minimum_fill_rate", "forecast"]
    paired_test = chosen.merge(paired, on=pair_keys, validate="m:1")
    pooled = summarize(paired_test, pooled_keys)
    fold_counts = paired_test.groupby(pooled_keys, as_index=False)["fold"].nunique().rename(
        columns={"fold": "paired_folds"},
    )
    return by_fold, pooled.merge(fold_counts, on=pooled_keys, validate="1:1")


def replay(inputs: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    paths = {label: inputs / name for label, name in (
        ("calibration", "calibration.csv.gz"), ("predictions", "predictions.csv.gz"),
    )}
    cal, test = (
        pd.read_csv(path, parse_dates=["week_start"], float_precision="round_trip")
        for path in paths.values()
    )
    folds = set(range(1, 7))
    if set(cal["fold"]) != folds or set(test["fold"]) != folds or test.duplicated(["id", "week_start"]).any():
        raise ValueError("Expected six distinct test windows and their separate saved calibration evidence")
    all_weeks = pd.DatetimeIndex(test["week_start"].unique()).sort_values()
    if not all_weeks.equals(pd.date_range(all_weeks.min(), periods=6 * 26, freq="7D")):
        raise ValueError("Test evidence must cover six nonoverlapping 26-week blocks")
    weekly, tuning, pretest, fixed_trace = [], [], [], []
    for fold in sorted(folds):
        cal_dates, cal_ids, cal_grid = make_grid(cal.loc[cal["fold"].eq(fold)], 16)
        test_dates, test_ids, test_grid = make_grid(test.loc[test["fold"].eq(fold)], 26)
        if cal_ids != test_ids or cal_dates[-1] + pd.Timedelta(weeks=1) != test_dates[0]:
            raise ValueError(f"Fold {fold}: calibration/test items or temporal ordering differ")
        weekly.extend(weekly_records(fold, test_dates, test_ids, cal_grid, test_grid, cal_grid["actual"], fixed_trace))
        fit_grid = {name: values[:BUFFER_FIT_WEEKS] for name, values in cal_grid.items()}
        tune_grid = {name: values[BUFFER_FIT_WEEKS:] for name, values in cal_grid.items()}
        tuning.extend(weekly_records(fold, cal_dates[BUFFER_FIT_WEEKS:], cal_ids, fit_grid, tune_grid, fit_grid["actual"]))
        pretest.extend(weekly_records(fold, test_dates, test_ids, fit_grid, test_grid, cal_grid["actual"]))
    weekly = pd.concat(weekly, ignore_index=True)
    frontier = summarize(weekly, GROUP)
    selected, paired = evaluate_pretest_selection(pd.concat(tuning, ignore_index=True), pd.concat(pretest, ignore_index=True))
    tables = {
        "weekly": weekly,
        "by_fold": summarize(weekly, ["fold", *GROUP]),
        "frontier": frontier,
        "test_frontier_thresholds": match_service(frontier),
        "pretest_selection": selected,
        "pretest_paired": paired,
        "trace_fixed": pd.concat(fixed_trace, ignore_index=True),
    }
    hashes = {}
    for label, path in paths.items():
        with path.open("rb") as handle:
            hashes[label] = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "evidence_sha256": hashes, "leads": list(LEADS), "residual_taus": list(LEVELS),
        "fixed_comparison_tau": FIXED_TAU, "service_thresholds": list(FILL_THRESHOLDS), "warmup_weeks": WARMUP,
        "pretest_buffer_fit_weeks": BUFFER_FIT_WEEKS, "pretest_policy_selection_weeks": 16 - BUFFER_FIT_WEEKS,
        "cost_profiles_holding_shortage_price_ratios": {name: list(ratios) for name, ratios in PROFILES.items()},
        "test_weeks": 26, "scored_weeks_per_fold": 26 - WARMUP, "series_per_fold": 100,
        "calibration_weeks": 16, "forecast_step": 1,
        "demand": "Recorded M5 sales used as a demand proxy in a lost-sales scenario.",
    }
    return tables, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/probabilistic_backtest")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/inventory_replay")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    tables, metadata = replay(args.inputs)
    filenames = {name: f"{name}.csv" for name in tables}
    filenames["trace_fixed"] += ".gz"
    if args.verify:
        if json.loads((args.outdir / "metadata.json").read_text(encoding="utf-8")) != metadata:
            raise ValueError("Replay input fingerprints or configuration changed")
        for name, frame in tables.items():
            if "week_start" in frame.columns:
                frame = frame.assign(week_start=frame["week_start"].dt.strftime("%Y-%m-%d"))
            saved = pd.read_csv(args.outdir / filenames[name], float_precision="round_trip")
            pd.testing.assert_frame_equal(frame, saved, check_dtype=False, rtol=1e-10, atol=1e-10)
    else:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            compression = {"method": "gzip", "mtime": 0} if name == "trace_fixed" else None
            frame.to_csv(args.outdir / filenames[name], index=False, compression=compression)
        (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Replayed {len(tables['by_fold'])} fold/policy combinations using saved predictions.")


if __name__ == "__main__":
    main()
