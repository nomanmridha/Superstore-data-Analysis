import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def monthly_sales_series(df: pd.DataFrame) -> pd.Series:
    ts = (
        df.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .asfreq("ME")
    )
    return ts


def forecast_sales(df: pd.DataFrame, periods: int = 6):
    ts = monthly_sales_series(df).dropna()

    # Validation split (last 6 months)
    if len(ts) < 18:
        # too short to split reliably; fallback
        train = ts
        test = None
    else:
        train = ts.iloc[:-6]
        test = ts.iloc[-6:]

    # Fit on train for MAPE
    if len(train) < 24:
        model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    else:
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=12
        ).fit()

    if test is not None:
        pred_test = model.forecast(6)
        mape = float(np.mean(np.abs((test - pred_test) / test)) * 100)
    else:
        mape = float("nan")

    # Fit on full data for future forecast
    if len(ts) < 24:
        final_model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
    else:
        final_model = ExponentialSmoothing(
            ts, trend="add", seasonal="add", seasonal_periods=12
        ).fit()

    fc = final_model.forecast(periods)

    # Approximate 95% CI from residual std
    resid = final_model.resid
    std = float(np.std(resid.dropna()))
    upper = fc + 1.96 * std
    lower = fc - 1.96 * std

    return ts, fc, lower, upper, mape
