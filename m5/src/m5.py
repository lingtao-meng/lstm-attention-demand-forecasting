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


def select_series(
    sales: pd.DataFrame, n_series: int = 100, seed: int = 42,
    selection_days: list[str] | None = None,
) -> pd.DataFrame:
    """Top 5 per store plus random; optionally score/qualify using history only.

    ``selection_days`` limits both the ranking and the nonzero eligibility
    filter. Other daily columns are retained for subsequent model evaluation.
    Omitting it preserves the original full-period sample.
    """
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    if selection_days is not None:
        if not selection_days or not set(selection_days).issubset(d_cols):
            raise ValueError("selection_days must be a nonempty list of available daily columns")
        d_cols = selection_days
    total = sales[d_cols].sum(axis=1)
    sales = sales.assign(total=total)
    if selection_days is not None:
        sales = sales[sales["total"] > 0]

    top = sales.sort_values("total", ascending=False).groupby("store_id", observed=True).head(5)
    remaining = sales.drop(index=top.index)
    remaining = remaining[remaining["total"] > 0]
    n_random = max(0, n_series - len(top))
    random = remaining.sample(n=min(n_random, len(remaining)), random_state=seed)
    selected = pd.concat([top, random], ignore_index=True)
    return selected.drop(columns=["total"])


def selection_history(
    sales: pd.DataFrame, calendar: pd.DataFrame, *, holdout: int = 26,
    gap_weeks: int = 16, test_start: str | pd.Timestamp | None = None,
) -> tuple[list[str], dict]:
    """Select daily columns strictly before an independent calibration block.

    Week boundaries come from the supplied retail calendar and are restricted
    to days actually present in the sales table, excluding incomplete weeks.
    """
    d_cols = {c for c in sales.columns if c.startswith("d_")}
    available = calendar[calendar["d"].isin(d_cols)].copy()
    available["date"] = pd.to_datetime(available["date"])
    complete = available.groupby("wm_yr_wk")["date"].nunique()
    available = available[available["wm_yr_wk"].isin(complete[complete == 7].index)]
    starts = available.groupby("wm_yr_wk")["date"].min().sort_values()
    if gap_weeks < 1 or holdout < 1:
        raise ValueError("holdout and gap_weeks must be positive")
    if test_start is None:
        if len(starts) <= holdout + gap_weeks:
            raise ValueError("Not enough complete retail weeks for selection/calibration/test")
        test_date = starts.iloc[-holdout]
    else:
        test_date = pd.Timestamp(test_start)
        if test_date not in set(starts):
            raise ValueError("test_start must be the start of a complete retail week")
    before_test = starts[starts < test_date]
    if len(before_test) <= gap_weeks:
        raise ValueError("Not enough history before calibration")
    calibration_start = before_test.iloc[-gap_weeks]
    history = available[available["date"] < calibration_start].sort_values("date")
    days = history["d"].tolist()
    return days, {
        "selection_start": str(history["date"].min().date()),
        "selection_end": str(history["date"].max().date()),
        "calibration_start": str(calibration_start.date()),
        "test_start": str(test_date.date()),
        "selection_days": len(days),
        "selection_gap_weeks": gap_weeks,
    }


def weekly_panel(
    selected: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Melt daily sales, aggregate to weeks, and attach calendar/price features."""
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    # 只保留 calendar 里真正有 7 天的零售周；最后一周常只有 2 天，不能按整周评价。
    days_per_week = calendar.groupby("wm_yr_wk")["date"].nunique()
    complete_weeks = days_per_week[days_per_week == 7].index
    calendar = calendar[calendar["wm_yr_wk"].isin(complete_weeks)].copy()
    week_start = calendar.groupby("wm_yr_wk")["date"].min()
    calendar["week_start"] = calendar["wm_yr_wk"].map(week_start)

    d_cols = [c for c in selected.columns if c.startswith("d_")]
    long = selected.melt(id_vars=ID_COLS, value_vars=d_cols, var_name="d", value_name="sales")
    long = long.merge(
        calendar[["d", "date", "wm_yr_wk", "week_start", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]],
        on="d",
        how="left",
    )
    long["date"] = pd.to_datetime(long["date"])
    # 真实锚点取该 wm 周在 calendar 的第一天（M5 是周六），仅用 Period 做排序/展示。
    long["week"] = long["week_start"].dt.to_period("W")
    long["event"] = long["event_name_1"].notna().astype(int)
    long["snap"] = np.select(
        [long["state_id"].eq("CA"), long["state_id"].eq("TX"), long["state_id"].eq("WI")],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]],
        default=0,
    )

    weekly = (
        long.groupby(ID_COLS + ["wm_yr_wk"], observed=True, as_index=False)
        .agg(sales=("sales", "sum"), event=("event", "max"), snap=("snap", "max"),
             week=("week", "first"), week_start=("week_start", "first"),
             n_days=("date", "nunique"))
        .sort_values(["id", "week"])
        .reset_index(drop=True)
    )
    # 只保留实际有 7 天销量的零售周；末尾 2 天不能按整周评价。
    weekly = weekly[weekly["n_days"] == 7].reset_index(drop=True)

    if prices is not None and len(prices):
        price_weekly = (
            prices.groupby(["store_id", "item_id", "wm_yr_wk"], observed=True, as_index=False)["sell_price"]
            .mean()
            .rename(columns={"sell_price": "price"})
        )
        weekly = weekly.merge(
            price_weekly, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="one_to_one"
        )
        # 只用过去已知价格：ffill 不引入未来；前几周缺失就留 NaN，交给模型处理
        weekly["price"] = weekly.groupby("id", observed=True)["price"].ffill()

    week_order = {week: idx for idx, week in enumerate(sorted(weekly["week"].unique()))}
    weekly["week_index"] = weekly["week"].map(week_order)
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month
    weekly["year"] = weekly["week_start"].dt.year
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
        # 只用当前周之前的价格均值，避免测试期价格反过来改变历史特征
        past_mean = price_group.transform(lambda s: s.expanding().mean().shift(1))
        df["price_rel"] = (df["price"] / past_mean).replace([np.inf, -np.inf], np.nan)
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
