# Streamlit dashboard for Superstore BI + Analytics project

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# Page config (title, layout)
# -----------------------------
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.write("✅ Streamlit loaded successfully.")
st.write("Python:", __import__("sys").version)


st.title("📊 Superstore Analytics Dashboard")
st.caption("BI + Customer Segmentation + Forecasting (Analysis-ready dataset)")

# -----------------------------
# Data loading (cached)
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Safety: ensure dates are datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    # Some datasets may contain missing numeric values; keep them as NaN
    # (KPIs will handle it safely).
    return df


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_PATH = ROOT / "data" / "processed" / "superstore_processed.csv"
df = load_data(str(DATA_PATH))


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

# Date range filter
min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

date_range = st.sidebar.date_input(
    "Order Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Category filter
all_categories = sorted(df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Category",
    options=all_categories,
    default=all_categories
)

# Region filter
all_regions = sorted(df["Region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Region",
    options=all_regions,
    default=all_regions
)

# Segment filter (this is the original retail segment, not RFM)
all_segments = sorted(df["Segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Customer Segment (Retail)",
    options=all_segments,
    default=all_segments
)

# -----------------------------
# Apply filters
# -----------------------------
filtered = df.copy()

filtered = filtered[
    (filtered["Order Date"] >= start_date) &
    (filtered["Order Date"] <= end_date)
]

if selected_categories:
    filtered = filtered[filtered["Category"].isin(selected_categories)]

if selected_regions:
    filtered = filtered[filtered["Region"].isin(selected_regions)]

if selected_segments:
    filtered = filtered[filtered["Segment"].isin(selected_segments)]

# -----------------------------
# KPI calculations
# -----------------------------
# NOTE: Use nan-safe sums (skipna=True is default)
total_sales = filtered["Sales"].sum()
total_profit = filtered["Profit"].sum()
profit_margin = (total_profit / total_sales * 100) if total_sales else np.nan
avg_discount = filtered["Discount"].mean()
orders = filtered["Order ID"].nunique()
customers = filtered["Customer ID"].nunique()

# -----------------------------
# KPI display
# -----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Total Sales", f"{total_sales:,.0f}")
k2.metric("Total Profit", f"{total_profit:,.0f}")
k3.metric("Profit Margin", f"{profit_margin:,.2f}%" if pd.notna(profit_margin) else "N/A")
k4.metric("Avg Discount", f"{avg_discount:.2%}" if pd.notna(avg_discount) else "N/A")
k5.metric("Orders", f"{orders:,}")
k6.metric("Customers", f"{customers:,}")

st.divider()

# -----------------------------
# Trend charts (monthly)
# -----------------------------
trend = (
    filtered
    .set_index("Order Date")
    .resample("ME")[["Sales", "Profit"]]
    .sum()
    .reset_index()
)

c1, c2 = st.columns(2)

with c1:
    fig_sales = px.line(trend, x="Order Date", y="Sales", title="Monthly Sales Trend")
    st.plotly_chart(fig_sales, use_container_width=True)

with c2:
    fig_profit = px.line(trend, x="Order Date", y="Profit", title="Monthly Profit Trend")
    st.plotly_chart(fig_profit, use_container_width=True)

st.divider()

# -----------------------------
# Category profitability
# -----------------------------
cat_perf = (
    filtered.groupby("Category", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values("Profit", ascending=False)
)

c3, c4 = st.columns(2)

with c3:
    fig_cat_profit = px.bar(cat_perf, x="Category", y="Profit", title="Profit by Category")
    st.plotly_chart(fig_cat_profit, use_container_width=True)

with c4:
    fig_cat_sales = px.bar(cat_perf, x="Category", y="Sales", title="Sales by Category")
    st.plotly_chart(fig_cat_sales, use_container_width=True)

st.divider()

# -----------------------------
# Discount vs Profit scatter (sampled)
# -----------------------------
scatter_sample = filtered[["Discount", "Profit"]].dropna()
if len(scatter_sample) > 3000:
    scatter_sample = scatter_sample.sample(3000, random_state=42)

fig_scatter = px.scatter(
    scatter_sample,
    x="Discount",
    y="Profit",
    title="Discount vs Profit (Sampled)",
    opacity=0.5
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------
# Raw data preview (optional)
# -----------------------------
with st.expander("Show filtered data (first 200 rows)"):
    st.dataframe(filtered.head(200))
