"""M5 data loading, subset selection, weekly aggregation and features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAT_COLS = ["store_id", "cat_id", "dept_id", "item_id"]
LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
WINDOWS = [4, 8, 13, 26, 52]


def load_sales(path: Path) -> pd.DataFrame:
    """Read the wide sales table with compact dtypes (the file is ~120 MB)."""
    header = pd.read_csv(path, nrows=0)
    d_cols = [c for c in header.columns if c.startswith("d_")]
    dtypes = {c: "int16" for c in d_cols}
    dtypes.update({c: "category" for c in ID_COLS})
    return pd.read_csv(path, dtype=dtypes)


def select_series(sales: pd.DataFrame, n_series: int = 100, seed: int = 42) -> pd.DataFrame:
    """Top 5 series per store plus a random sample, so the panel is not all best-sellers."""
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    total = sales[d_cols].sum(axis=1)
    sales = sales.assign(total=total)

    top = sales.sort_values("total", ascending=False).groupby("store_id", observed=True).head(5)
    remaining = sales.drop(index=top.index)
    remaining = remaining[remaining["total"] > 0]
    n_random = max(0, n_series - len(top))
    random = remaining.sample(n=min(n_random, len(remaining)), random_state=seed)
    selected = pd.concat([top, random], ignore_index=True)
    return selected.drop(columns=["total"])


def weekly_panel(
    selected: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Melt daily sales, aggregate to weeks, and attach calendar/price features."""
    calendar = calendar.assign(week=pd.to_datetime(calendar["date"]).dt.to_period("W"))
    d_cols = [c for c in selected.columns if c.startswith("d_")]
    long = selected.melt(id_vars=ID_COLS, value_vars=d_cols, var_name="d", value_name="sales")
    long = long.merge(
        calendar[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]],
        on="d",
        how="left",
    )
    long["date"] = pd.to_datetime(long["date"])
    long["week"] = long["date"].dt.to_period("W")
    long["event"] = long["event_name_1"].notna().astype(int)
    long["snap"] = np.select(
        [long["state_id"].eq("CA"), long["state_id"].eq("TX"), long["state_id"].eq("WI")],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]],
        default=0,
    )

    weekly = (
        long.groupby(ID_COLS + ["week"], observed=True, as_index=False)
        .agg(sales=("sales", "sum"), event=("event", "max"), snap=("snap", "max"))
        .sort_values(["id", "week"])
        .reset_index(drop=True)
    )

    if prices is not None and len(prices):
        week_map = calendar[["wm_yr_wk", "week"]].drop_duplicates("wm_yr_wk")
        price_weekly = (
            prices.merge(week_map, on="wm_yr_wk", how="inner")
            .groupby(["store_id", "item_id", "week"], observed=True, as_index=False)["sell_price"]
            .mean()
            .rename(columns={"sell_price": "price"})
        )
        weekly = weekly.merge(price_weekly, on=["store_id", "item_id", "week"], how="left")
        weekly["price"] = weekly.groupby("id", observed=True)["price"].ffill().bfill()
        weekly["price"] = weekly["price"].fillna(weekly.groupby("id", observed=True)["price"].transform("median"))

    week_order = {week: idx for idx, week in enumerate(sorted(weekly["week"].unique()))}
    weekly["week_index"] = weekly["week"].map(week_order)
    weekly["week_of_year"] = weekly["week"].apply(lambda x: x.week)
    weekly["month"] = weekly["week"].apply(lambda x: x.start_time.month)
    weekly["year"] = weekly["week"].apply(lambda x: x.start_time.year)
    return weekly


def add_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Lagged sales, rolling statistics, price and calendar features."""
    df = weekly.sort_values(["id", "week"]).copy()
    if "week_of_year" not in df.columns:
        df["week_of_year"] = df["week"].apply(lambda x: x.week)
    if "month" not in df.columns:
        df["month"] = df["week"].apply(lambda x: x.start_time.month)
    if "year" not in df.columns:
        df["year"] = df["week"].apply(lambda x: x.start_time.year)
    if "week_index" not in df.columns:
        week_order = {week: idx for idx, week in enumerate(sorted(df["week"].unique()))}
        df["week_index"] = df["week"].map(week_order)
    for flag in ("event", "snap"):
        if flag not in df.columns:
            df[flag] = 0
    grouped = df.groupby("id", observed=True)["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in WINDOWS:
        shifted = grouped.shift(1)
        df[f"roll_mean_{window}"] = shifted.groupby(df["id"], observed=True).transform(
            lambda s: s.rolling(window).mean()
        )
        df[f"roll_std_{window}"] = shifted.groupby(df["id"], observed=True).transform(
            lambda s: s.rolling(window).std()
        )

    df["zero_share_13"] = grouped.shift(1).groupby(df["id"], observed=True).transform(
        lambda s: (s == 0).rolling(13).mean()
    )
    df["ewm_mean_8"] = grouped.shift(1).groupby(df["id"], observed=True).transform(
        lambda s: s.ewm(span=8, adjust=False).mean()
    )
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    if "price" in df.columns:
        price_group = df.groupby("id", observed=True)["price"]
        df["price_rel"] = df["price"] / price_group.transform("mean")
        df["price_change"] = price_group.pct_change().replace([np.inf, -np.inf], np.nan)

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [f"lag_{lag}" for lag in LAGS]
    cols += [f"roll_mean_{w}" for w in WINDOWS]
    cols += [f"roll_std_{w}" for w in WINDOWS]
    cols += ["week_of_year", "week_sin", "week_cos", "month", "year", "week_index", "event", "snap"]
    cols += ["zero_share_13", "ewm_mean_8"]
    for optional in ("price", "price_rel", "price_change"):
        if optional in df.columns:
            cols.append(optional)
    cols += [c for c in CAT_COLS if c in df.columns]
    return cols


def add_baselines(weekly: pd.DataFrame) -> pd.DataFrame:
    """Naive, seasonal-naive and 4-week moving-average forecasts (no leakage)."""
    df = weekly.sort_values(["id", "week"]).copy()
    if "week_of_year" not in df.columns:
        df["week_of_year"] = df["week"].apply(lambda x: x.week)
    if "month" not in df.columns:
        df["month"] = df["week"].apply(lambda x: x.start_time.month)
    if "year" not in df.columns:
        df["year"] = df["week"].apply(lambda x: x.start_time.year)
    if "week_index" not in df.columns:
        week_order = {week: idx for idx, week in enumerate(sorted(df["week"].unique()))}
        df["week_index"] = df["week"].map(week_order)
    for flag in ("event", "snap"):
        if flag not in df.columns:
            df[flag] = 0
    grouped = df.groupby("id", observed=True)["sales"]
    df["naive"] = grouped.shift(1)
    df["seasonal"] = grouped.shift(52)
    df["ma4"] = grouped.transform(lambda s: s.shift(1).rolling(4).mean())
    return df


def time_split(df: pd.DataFrame, holdout: int = 26):
    weeks = sorted(df["week"].unique())
    holdout_start = weeks[-holdout]
    train = df[df["week"] < holdout_start].copy()
    test = df[df["week"] >= holdout_start].copy()
    return train, test, holdout_start
