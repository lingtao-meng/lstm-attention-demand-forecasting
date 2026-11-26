"""Paired six-fold replay under late receipts, box sizes and extra target stock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from inventory_replay import PROFILES, WARMUP
from src.inventory import simulate_inventory

SEED = 2026112601
DRAWS = 40
SCENARIOS = (("on_time", 0.0), ("late_10", .10), ("late_30", .30), ("late_10_with_disruption", .10))
CASE_PACKS = (1, 6, 12)
EXTRA_WEEKS = (0.0, 0.5)
SERVICE_TARGETS = (.90, .93, .95)
DISRUPTED_ORDER_WEEKS = (8, 15)  # 1-based inside each fold.
GROUP = ["lead", "scenario", "case_pack", "extra_weeks", "forecast"]
COST_COLUMNS = [f"cost_{name}" for name in PROFILES]


def delay_calendars(fold: int, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Use common random numbers for every policy at the same order opportunity."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, fold]))
    uniforms = rng.random((DRAWS, *shape))
    calendars = {name: (uniforms < probability).astype(np.int8) for name, probability in SCENARIOS}
    calendars["late_10_with_disruption"][:, np.array(DISRUPTED_ORDER_WEEKS) - 1, :] = 1
    return calendars


def check_inventory(demand: np.ndarray, state: dict, pack: int) -> None:
    np.testing.assert_array_equal(state["fulfilled"] + state["lost"], demand)
    np.testing.assert_array_equal(state["opening_stock"] + state["arrivals"],
                                  state["fulfilled"] + state["closing_stock"])
    np.testing.assert_array_equal(state["closing_pipeline"][1:],
                                  state["closing_pipeline"][:-1] + state["order"][1:] - state["arrivals"][1:])
    if (state["order"] % pack).any() or (state["closing_pipeline"] < 0).any():
        raise AssertionError("Box constraint or outstanding-stock conservation failed")


def replay(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, lead, forecast), group in trace.groupby(["fold", "lead", "forecast"], sort=True):
        panel = group.set_index(["week_start", "id"]).sort_index()

        def matrix(column):
            return panel[column].unstack().to_numpy()

        demand, target, price = (matrix(col) for col in ("demand", "order_up_to", "price"))
        if demand.shape != (26, 100) or not np.isfinite(price).all() or (price <= 0).any():
            raise ValueError(f"Invalid replay panel for {fold=}, {lead=}, {forecast=}")
        np.testing.assert_array_equal(matrix("warmup"), np.broadcast_to(
            (np.arange(len(demand)) < WARMUP)[:, None], demand.shape))
        initial = matrix("opening_stock")[0]
        calendars = delay_calendars(fold, demand.shape)
        measured = slice(WARMUP, None)
        baseline = simulate_inventory(demand, target, initial, lead)
        for field, values in baseline.items():
            np.testing.assert_array_equal(values, matrix(field))

        for (scenario, delays), pack, extra in product(calendars.items(), CASE_PACKS, EXTRA_WEEKS):
            buffered = np.ceil(target * (1 + extra / (lead + 1))).astype(np.int64)
            for draw, calendar in enumerate(delays):
                state = simulate_inventory(demand, buffered, initial, lead, pack, extra_delay=calendar)
                check_inventory(demand, state, pack)
                stock, lost = (state[key][measured] for key in ("closing_stock", "lost"))
                priced_stock = (price[measured] * stock).sum()
                priced_lost = (price[measured] * lost).sum()
                rows.append({
                    "fold": fold, "draw": draw, "lead": lead, "scenario": scenario, "forecast": forecast,
                    "case_pack": pack, "extra_weeks": extra, "series_weeks": stock.size,
                    "demand": demand[measured].sum(), "fulfilled": state["fulfilled"][measured].sum(),
                    "lost": lost.sum(), "stock": stock.sum(), "stockouts": (lost > 0).sum(),
                    **{f"cost_{name}": h * priced_stock + p * priced_lost for name, (h, p) in PROFILES.items()},
                })
    expected_rows = 6 * 2 * 2 * len(SCENARIOS) * len(CASE_PACKS) * len(EXTRA_WEEKS) * DRAWS
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(rows)}")
    return pd.DataFrame(rows)


def summarize(replays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quantities = ["series_weeks", "demand", "fulfilled", "lost", "stock", "stockouts", *COST_COLUMNS]
    pooled = replays.groupby([*GROUP, "draw"], as_index=False)[quantities].sum()
    pooled["fill"] = pooled["fulfilled"] / pooled["demand"]
    pooled["stockout_rate"] = pooled["stockouts"] / pooled["series_weeks"]
    pooled["mean_stock"] = pooled["stock"] / pooled["series_weeks"]
    for name in COST_COLUMNS:
        pooled[name] /= pooled["series_weeks"]
    for level in SERVICE_TARGETS:
        pooled[f"meets_{int(level * 100)}"] = pooled["fill"].ge(level)
    average_fields = ["fill", "stockout_rate", "mean_stock", *COST_COLUMNS,
                      *(f"meets_{int(level * 100)}" for level in SERVICE_TARGETS)]
    summary = pooled.groupby(GROUP, as_index=False)[average_fields].mean()
    pair_keys = [key for key in GROUP if key != "forecast"]
    paired = pooled.pivot(index=[*pair_keys, "draw"], columns="forecast", values=["cost_balanced", "fill"])
    improvements = pd.DataFrame({
        "cost_saving": paired[("cost_balanced", "naive")] - paired[("cost_balanced", "q50")],
        "fill_delta": paired[("fill", "q50")] - paired[("fill", "naive")],
    }).reset_index()
    improvements["joint_improvement"] = (improvements["cost_saving"] > 0) & (improvements["fill_delta"] >= 0)
    paired_summary = improvements.groupby(pair_keys, as_index=False).agg(
        cost_saving=("cost_saving", "mean"), cost_saving_p05=("cost_saving", lambda x: x.quantile(.05)),
        cost_saving_p95=("cost_saving", lambda x: x.quantile(.95)), fill_delta=("fill_delta", "mean"),
        both_better_share=("joint_improvement", "mean"),
    )
    return summary, paired_summary, improvements


def plot_service(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    from figure_style import format_figure

    names = [label for label, _ in SCENARIOS]
    labels = ["On time", "10% late", "30% late", "10% +\ndisruption"]
    series = summary.query("forecast == 'q50' and case_pack == 1")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True, layout="constrained")
    for lead, ax in enumerate(axes, start=1):
        for extra, color, marker, line, label in (
            (0.0, "#575c60", "s", "--", "Original target"),
            (0.5, "#315c77", "o", "-", "Higher stock target"),
        ):
            rows = series.loc[series.lead.eq(lead) & series.extra_weeks.eq(extra)].set_index("scenario").loc[names]
            ax.plot(np.arange(len(names)), 100 * rows["fill"], color=color, marker=marker,
                    linestyle=line, markersize=5, label=label)
        ax.axhline(95, color="#8a6147", linewidth=.8, linestyle=":", label="95% service target")
        ax.set(xticks=np.arange(len(names)), xticklabels=labels, ylim=(78, 101),
               title=f"{'AB'[lead - 1]}  Nominal lead time: {lead} week" + ("s" if lead > 1 else ""))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Demand units filled (%)")
    axes[0].legend(fontsize=9, loc="lower left")
    format_figure(fig)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "results/inventory_replay/trace_fixed.csv.gz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/inventory_delay_stress")
    args = parser.parse_args()
    traces = replay(pd.read_csv(args.trace))
    summary, paired, draws = summarize(traces)
    by_fold = traces.groupby(["fold", *GROUP], as_index=False).mean(numeric_only=True).drop(columns="draw")
    args.outdir.mkdir(parents=True, exist_ok=True)
    for filename, table in (("summary", summary), ("paired", paired), ("paired_draws", draws),
                            ("by_fold", by_fold)):
        table.to_csv(args.outdir / f"{filename}.csv", index=False, float_format="%.8f")
    metadata = {"trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(), "seed": SEED,
                "draws_per_fold": DRAWS, "folds": 6, "warmup_weeks": WARMUP,
                "scenario_late_probabilities": dict(SCENARIOS), "delay_when_late_weeks": 1,
                "disrupted_order_weeks": DISRUPTED_ORDER_WEEKS, "cases": CASE_PACKS,
                "extra_target_weeks": EXTRA_WEEKS, "service_targets": SERVICE_TARGETS,
                "score_rows_per_policy_and_draw": 6 * (26 - WARMUP) * 100, "cost_profiles": PROFILES}
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_service(summary, args.outdir / "service_under_delay.png")
    print(f"Evaluated {len(traces):,} paired fold, policy and receipt-calendar runs.")


if __name__ == "__main__":
    main()
