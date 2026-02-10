"""Lean RFM segmentation utilities for the Superstore Analytics project.

This intentionally keeps the modeling simple:
- RFM features per Customer ID
- KMeans clustering with fixed k (default 4)
- Cluster profiling + human-readable labels

Designed for a Master's course project (Advanced Analytics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency/Frequency/Monetary per customer.

    Required columns: 'Customer ID', 'Order Date', 'Order ID', 'Sales'
    Recency: days since last purchase (relative to max date + 1 day)
    Frequency: number of unique orders
    Monetary: total Sales
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

    d = df.copy()
    d["Order Date"] = pd.to_datetime(d["Order Date"], errors="coerce")
    d = d.dropna(subset=["Order Date"])
    if d.empty:
        return pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

    snapshot_date = d["Order Date"].max() + pd.Timedelta(days=1)

    rfm = d.groupby("Customer ID").agg(
        Recency=("Order Date", lambda x: int((snapshot_date - x.max()).days)),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
    )

    # guardrails
    rfm["Monetary"] = rfm["Monetary"].clip(lower=0)
    rfm["Frequency"] = rfm["Frequency"].clip(lower=0)
    rfm["Recency"] = rfm["Recency"].clip(lower=0)

    return rfm


def run_kmeans(rfm: pd.DataFrame, k: int = 4, random_state: int = 42) -> pd.DataFrame:
    """Cluster RFM with KMeans (fixed k).

    Notes:
    - Uses log1p(Monetary) to reduce skew
    - StandardScaler for comparable feature scales
    """
    if rfm is None or rfm.empty:
        out = rfm.copy() if rfm is not None else pd.DataFrame()
        out["Cluster"] = []
        return out

    feats = rfm[["Recency", "Frequency", "Monetary"]].copy()
    feats["Monetary"] = np.log1p(feats["Monetary"])

    X = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
    out = rfm.copy()
    out["Cluster"] = km.fit_predict(X).astype(int)
    return out


def profile_clusters(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    """Return mean Recency/Frequency/Monetary and customer count per cluster."""
    if rfm_clustered is None or rfm_clustered.empty:
        return pd.DataFrame(columns=["Recency", "Frequency", "Monetary", "Customers"])

    return (
        rfm_clustered.groupby("Cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            Customers=("Cluster", "size"),
        )
        .round(2)
    )


def label_clusters(profile: pd.DataFrame) -> pd.DataFrame:
    """Add a simple business-friendly segment label per cluster.

    Heuristic based on ranks:
    - Recency: lower is better
    - Frequency, Monetary: higher is better
    """
    if profile is None or profile.empty:
        out = profile.copy() if profile is not None else pd.DataFrame()
        out["Segment"] = []
        return out

    p = profile.copy()

    # ranks: 1 = best
    r_rank = p["Recency"].rank(ascending=True, method="average")
    f_rank = p["Frequency"].rank(ascending=False, method="average")
    m_rank = p["Monetary"].rank(ascending=False, method="average")

    score = (f_rank + m_rank) - r_rank  # higher => better
    best = score.idxmax()
    worst = score.idxmin()

    labels = {c: "Loyal" for c in p.index}
    labels[best] = "Champions"
    labels[worst] = "Lost / At Risk"

    # optional: identify a 'Big Spenders' group (top Monetary but not best/worst)
    top_m = m_rank.idxmin()  # rank 1 is smallest numeric value
    if top_m not in (best, worst):
        labels[top_m] = "Big Spenders"

    p["Segment"] = [labels[c] for c in p.index]
    return p
