import pandas as pd

# Requires: pip install statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def monthly_sales_series(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate monthly sales as a time series indexed by month-end.
    """
    ts = (
        df.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .asfreq("ME")
    )
    return ts


def forecast_sales(df: pd.DataFrame, periods: int = 6):
    """
    Forecast future monthly sales using Holt-Winters (additive trend + seasonality).

    Returns:
        hist: pd.Series (monthly sales)
        fc:   pd.Series (future forecast)
    """
    ts = monthly_sales_series(df).dropna()

    # Safety: if not enough history for seasonality, fall back to non-seasonal model
    if len(ts) < 24:  # < 2 seasons is weak for seasonal=12
        model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
    else:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
        ).fit()

    fc = model.forecast(periods)
    return ts, fc
