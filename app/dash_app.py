"""
Dash dashboard for Superstore Analytics project

Run:
    .\venv312\Scripts\activate
    python .\app\dash_app.py

Open:
    http://127.0.0.1:8050
"""

from pathlib import Path
from forecasting import forecast_sales

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output


# -----------------------------
# Paths + Data
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_PATH = ROOT / "data" / "processed" / "superstore_processed.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Ensure required columns exist (fail fast with a clear message)
required_cols = [
    "Order Date", "Ship Date", "Category", "Region", "Segment",
    "Sales", "Profit", "Discount", "Order ID", "Customer ID"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        "Your CSV is missing required columns:\n"
        + "\n".join(missing)
        + "\n\nCheck your processed dataset columns vs. expected schema."
    )

# Datetime parsing
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
df = df.dropna(subset=["Order Date"])

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

all_categories = sorted(df["Category"].dropna().unique().tolist())
all_regions = sorted(df["Region"].dropna().unique().tolist())
all_segments = sorted(df["Segment"].dropna().unique().tolist())


# -----------------------------
# Styling
# -----------------------------
PAGE_STYLE = {"fontFamily": "Arial", "margin": "18px"}

KPI_CARD_STYLE = {
    "border": "1px solid #ddd",
    "borderRadius": "10px",
    "padding": "10px",
    "backgroundColor": "#fafafa",
    "minHeight": "70px",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
}

LABEL_STYLE = {"fontSize": 12, "color": "#666"}
VALUE_STYLE = {"fontSize": 20, "fontWeight": "bold"}


def _format_int(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "N/A"


def _format_pct(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:,.2f}%"


def _kpi_block(label: str, value: str):
    return [
        html.Div(label, style=LABEL_STYLE),
        html.Div(value, style=VALUE_STYLE),
    ]


# -----------------------------
# App
# -----------------------------
app = Dash(__name__)
app.title = "Superstore Analytics Dashboard"

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.H1("📊 Superstore Analytics Dashboard"),
        html.P("BI + Customer Segmentation + Forecasting (Analysis-ready dataset)"),

        # -------- Filters --------
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                "gap": "14px",
                "marginBottom": "10px",
                "padding": "12px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
            },
            children=[
                html.Div([
                    html.Label("Order Date range"),
                    dcc.DatePickerRange(
                        id="date_range",
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        display_format="YYYY-MM-DD",
                    ),
                ]),
                html.Div([
                    html.Label("Category"),
                    dcc.Dropdown(
                        id="category_dd",
                        options=[{"label": c, "value": c} for c in all_categories],
                        value=all_categories,
                        multi=True,
                    ),
                ]),
                html.Div([
                    html.Label("Region"),
                    dcc.Dropdown(
                        id="region_dd",
                        options=[{"label": r, "value": r} for r in all_regions],
                        value=all_regions,
                        multi=True,
                    ),
                ]),
                html.Div([
                    html.Label("Retail Segment"),
                    dcc.Dropdown(
                        id="segment_dd",
                        options=[{"label": s, "value": s} for s in all_segments],
                        value=all_segments,
                        multi=True,
                    ),
                ]),
            ],
        ),

        # -------- KPI Row --------
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(6, 1fr)",
                "gap": "10px",
                "marginBottom": "12px",
            },
            children=[
                html.Div(id="kpi_sales", style=KPI_CARD_STYLE),
                html.Div(id="kpi_profit", style=KPI_CARD_STYLE),
                html.Div(id="kpi_margin", style=KPI_CARD_STYLE),
                html.Div(id="kpi_discount", style=KPI_CARD_STYLE),
                html.Div(id="kpi_orders", style=KPI_CARD_STYLE),
                html.Div(id="kpi_customers", style=KPI_CARD_STYLE),
            ],
        ),

        # -------- Charts --------
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                dcc.Graph(id="sales_trend"),
                dcc.Graph(id="profit_trend"),
                dcc.Graph(id="forecast_chart"),
            ],
        ),

        html.Div(style={"height": "8px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                dcc.Graph(id="profit_by_category"),
                dcc.Graph(id="sales_by_category"),
            ],
        ),

        html.Div(style={"height": "8px"}),

        dcc.Graph(id="discount_vs_profit"),

        # -------- Data preview --------
        html.H3("Filtered data preview (first 200 rows)"),
        dash_table.DataTable(
            id="data_table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "6px"},
        ),

        html.Div(
            style={"marginTop": "10px", "color": "#666", "fontSize": 12},
            children=[
                html.Span("Data source: "),
                html.Code(str(DATA_PATH)),
            ],
        ),
    ],
)


@app.callback(
    Output("kpi_sales", "children"),
    Output("kpi_profit", "children"),
    Output("kpi_margin", "children"),
    Output("kpi_discount", "children"),
    Output("kpi_orders", "children"),
    Output("kpi_customers", "children"),
    Output("sales_trend", "figure"),
    Output("profit_trend", "figure"),
    Output("profit_by_category", "figure"),
    Output("sales_by_category", "figure"),
    Output("discount_vs_profit", "figure"),
    Output("data_table", "data"),
    Output("data_table", "columns"),
    Output("forecast_chart", "figure"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("category_dd", "value"),
    Input("region_dd", "value"),
    Input("segment_dd", "value"),
)
def update_dashboard(start_date, end_date, categories, regions, segments):
    filtered = df.copy()
# Apply filters FIRST
    if start_date:
        filtered = filtered[filtered["Order Date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["Order Date"] <= pd.to_datetime(end_date)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if segments:
        filtered = filtered[filtered["Segment"].isin(segments)]

    # THEN forecast using filtered data
    hist, fc = forecast_sales(filtered, periods=6)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Actual Sales"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=fc.index, y=fc.values,
        mode="lines", name="Forecast",
        line=dict(dash="dash")
    ))
    fig_forecast.update_layout(title="Sales Forecast (Next 6 Months)")

    
    if start_date:
        filtered = filtered[filtered["Order Date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["Order Date"] <= pd.to_datetime(end_date)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if segments:
        filtered = filtered[filtered["Segment"].isin(segments)]

    # KPIs
    total_sales = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales else np.nan
    avg_discount = filtered["Discount"].mean()
    orders = filtered["Order ID"].nunique()
    customers = filtered["Customer ID"].nunique()

    kpi_sales = _kpi_block("Total Sales", _format_int(total_sales))
    kpi_profit = _kpi_block("Total Profit", _format_int(total_profit))
    kpi_margin = _kpi_block("Profit Margin", _format_pct(profit_margin))
    kpi_discount = _kpi_block("Avg Discount", "N/A" if pd.isna(avg_discount) else f"{avg_discount:.2%}")
    kpi_orders = _kpi_block("Orders", _format_int(orders))
    kpi_customers = _kpi_block("Customers", _format_int(customers))

    # Trends (monthly)
    trend = (
        filtered.set_index("Order Date")[["Sales", "Profit"]]
        .resample("ME")
        .sum()
        .reset_index()
    )

    fig_sales = px.line(trend, x="Order Date", y="Sales", title="Monthly Sales Trend")
    fig_profit = px.line(trend, x="Order Date", y="Profit", title="Monthly Profit Trend")

    # Category performance
    cat_perf = (
        filtered.groupby("Category", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values("Profit", ascending=False)
    )
    fig_cat_profit = px.bar(cat_perf, x="Category", y="Profit", title="Profit by Category")
    fig_cat_sales = px.bar(cat_perf, x="Category", y="Sales", title="Sales by Category")

    # Discount vs Profit (sample)
    scatter_sample = filtered[["Discount", "Profit"]].dropna()
    if len(scatter_sample) > 3000:
        scatter_sample = scatter_sample.sample(3000, random_state=42)

    fig_scatter = px.scatter(
        scatter_sample, x="Discount", y="Profit",
        title="Discount vs Profit (Sampled)", opacity=0.5
    )

    # Data table
    preview = filtered.head(200).copy()
    columns = [{"name": c, "id": c} for c in preview.columns]
    data = preview.to_dict("records")

    return (
        kpi_sales, kpi_profit, kpi_margin, kpi_discount, kpi_orders, kpi_customers,
        fig_sales, fig_profit,
        fig_cat_profit, fig_cat_sales,
        fig_scatter,
        data, columns,
        fig_forecast
    )



if __name__ == "__main__":
    # Dash 4.x: run_server is obsolete, use app.run
    app.run(debug=True, host="127.0.0.1", port=8050)
