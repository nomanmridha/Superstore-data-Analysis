from __future__ import annotations

import numpy as np
import pandas as pd


def _monthly_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Aggregate to month-end series."""
    s = (
        df.set_index("Order Date")[value_col]
        .resample("ME")
        .sum()
        .dropna()
    )
    s.index = pd.to_datetime(s.index)
    return s


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def forecast_metric(
    df: pd.DataFrame,
    value_col: str = "Sales",
    periods: int = 6,
    ci: float = 0.95,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    """
    Simple seasonal-naive forecast:
      - Forecast each future month as the mean of same-month values from history.
      - Adds a wide CI using residual std.
    Returns:
      hist, fc, lower, upper, metrics_dict
    metrics keys:
      mape, rmse, forecast_sum, last6_actual_sum, growth_pct
    """
    if "Order Date" not in df.columns:
        raise ValueError("forecast_metric expects column 'Order Date' in df.")
    if value_col not in df.columns:
        raise ValueError(f"forecast_metric expects '{value_col}' in df.")

    hist = _monthly_series(df, value_col=value_col)

    # Guard: too few points
    if len(hist) < 8:
        idx = pd.date_range(hist.index.max() + pd.offsets.MonthEnd(1), periods=periods, freq="ME")
        fc = pd.Series([hist.mean() if len(hist) else 0.0] * periods, index=idx)
        lower = fc.copy()
        upper = fc.copy()
        metrics = {
            "mape": np.nan,
            "rmse": np.nan,
            "forecast_sum": float(fc.sum()),
            "last6_actual_sum": float(hist.tail(min(6, len(hist))).sum()) if len(hist) else np.nan,
            "growth_pct": np.nan,
        }
        return hist, fc, lower, upper, metrics

    # Seasonal naive by month-of-year mean
    month_means = hist.groupby(hist.index.month).mean()

    future_idx = pd.date_range(hist.index.max() + pd.offsets.MonthEnd(1), periods=periods, freq="ME")
    fc_vals = [float(month_means.loc[d.month]) for d in future_idx]
    fc = pd.Series(fc_vals, index=future_idx)

    # Backtest last 6 months
    back_n = min(6, len(hist) - 1)
    y_true = hist.tail(back_n).values
    # "predict" those months using the same seasonal rule
    y_pred = np.array([float(month_means.loc[d.month]) for d in hist.tail(back_n).index], dtype=float)

    mape = _mape(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)

    # CI using residual std from backtest (normal approx)
    resid = y_true - y_pred
    resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    # z for ~95% (kept simple)
    z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96
    lower = fc - z * resid_std
    upper = fc + z * resid_std

    last6_actual_sum = float(hist.tail(6).sum())
    forecast_sum = float(fc.sum())
    growth_pct = float(((forecast_sum / last6_actual_sum) - 1.0) * 100.0) if last6_actual_sum != 0 else np.nan

    metrics = {
        "mape": float(mape),
        "rmse": float(rmse),
        "forecast_sum": forecast_sum,
        "last6_actual_sum": last6_actual_sum,
        "growth_pct": growth_pct,
    }
    return hist, fc, lower, upper, metrics


# Backward-compatible alias (your dash previously used this name)
def forecast_sales(df: pd.DataFrame, periods: int = 6):
    return forecast_metric(df, value_col="Sales", periods=periods)


def forecast_profit(df: pd.DataFrame, periods: int = 6):
    return forecast_metric(df, value_col="Profit", periods=periods)


def forecast_segment_metrics(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "Sales",
    periods: int = 6,
    min_months: int = 12,
) -> pd.DataFrame:
    """Per-segment forecast KPIs (lean + robust).

    Uses the same seasonal-naive logic as `forecast_metric` per segment.
    Does **not** silently fail all rows: if a segment errors, it records status.

    Columns:
      - <group_col>, months, forecast_sum, last6_actual_sum, growth_pct, mape, rmse, status
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in df.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in df.")

    rows = []
    for g, gdf in df.groupby(group_col):
        hist = _monthly_series(gdf, value_col=value_col)
        if len(hist) < min_months:
            rows.append({
                group_col: g,
                "months": int(len(hist)),
                "forecast_sum": np.nan,
                "last6_actual_sum": float(hist.tail(min(6, len(hist))).sum()) if len(hist) else np.nan,
                "growth_pct": np.nan,
                "mape": np.nan,
                "rmse": np.nan,
                "status": f"insufficient_history(<{min_months})",
            })
            continue

        try:
            _, _, _, _, metrics = forecast_metric(gdf, value_col=value_col, periods=periods)
            rows.append({
                group_col: g,
                "months": int(len(hist)),
                "forecast_sum": float(metrics.get("forecast_sum", np.nan)),
                "last6_actual_sum": float(metrics.get("last6_actual_sum", np.nan)),
                "growth_pct": float(metrics.get("growth_pct", np.nan)),
                "mape": float(metrics.get("mape", np.nan)),
                "rmse": float(metrics.get("rmse", np.nan)),
                "status": "ok",
            })
        except Exception as e:
            rows.append({
                group_col: g,
                "months": int(len(hist)),
                "forecast_sum": np.nan,
                "last6_actual_sum": float(hist.tail(min(6, len(hist))).sum()) if len(hist) else np.nan,
                "growth_pct": np.nan,
                "mape": np.nan,
                "rmse": np.nan,
                "status": f"error: {type(e).__name__}",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sort: largest forecast_sum first (NaNs last)
    return out.sort_values("forecast_sum", ascending=False, na_position="last").reset_index(drop=True)

def holdout_validation(
    df: pd.DataFrame,
    value_col: str = "Sales",
    test_months: int = 12,
) -> dict:
    """Simple 1-split holdout validation on monthly series.

    Train: all but last `test_months` months
    Test:  last `test_months` months
    Model: seasonal-naive using train month-of-year means

    Returns dict with: mae, rmse, mape, bias, train_months, test_months
    """
    hist = _monthly_series(df, value_col=value_col)

    if len(hist) < (test_months + 6):
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "bias": np.nan,
            "train_months": int(max(len(hist) - test_months, 0)),
            "test_months": int(min(test_months, len(hist))),
        }

    train = hist.iloc[:-test_months]
    test = hist.iloc[-test_months:]

    month_means = train.groupby(train.index.month).mean()
    pred = np.array([float(month_means.loc[d.month]) for d in test.index], dtype=float)

    y_true = test.values.astype(float)
    mae = float(np.nanmean(np.abs(y_true - pred)))
    rmse = _rmse(y_true, pred)
    mape = _mape(y_true, pred)
    bias = float(np.nanmean(pred - y_true))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "bias": round(bias, 2),
        "train_months": int(len(train)),
        "test_months": int(len(test)),
    }
