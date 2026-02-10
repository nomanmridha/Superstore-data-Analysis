"""
Dash dashboard for Superstore Analytics project

Run:
    .\venv312\Scripts\activate
    python .\app\dash_app.py

Open:
    http://127.0.0.1:8050
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output

from forecasting import forecast_metric, forecast_segment_metrics, holdout_validation
from rfm_model import compute_rfm, run_kmeans, profile_clusters, label_clusters


# -----------------------------
# Paths + Data
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_PATH = ROOT / "data" / "processed" / "superstore_processed.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

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

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
df = df.dropna(subset=["Order Date"])

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

all_categories = sorted(df["Category"].dropna().unique().tolist())
all_regions = sorted(df["Region"].dropna().unique().tolist())
all_segments = sorted(df["Segment"].dropna().unique().tolist())


# -----------------------------
# Styling + Helpers
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
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.0f}"
    except Exception:
        return "N/A"


def _format_float(x, nd=2) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "N/A"


def _format_pct(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{float(x):,.2f}%"


def _kpi_block(label: str, value: str):
    return [
        html.Div(label, style=LABEL_STYLE),
        html.Div(value, style=VALUE_STYLE),
    ]


# -----------------------------
# App + Layout
# -----------------------------
app = Dash(__name__)
app.title = "Superstore Analytics Dashboard"

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.H1("📊 Superstore Analytics Dashboard"),
        html.P("BI + Customer Segmentation + Forecasting (Analysis-ready dataset)"),

        # -------- Filters + Forecast Controls --------
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.2fr 1fr 1fr 1fr",
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

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginBottom": "10px"},
            children=[
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                    children=[
                        html.Label("Forecast metric"),
                        dcc.RadioItems(
                            id="forecast_metric",
                            options=[
                                {"label": "Sales", "value": "Sales"},
                                {"label": "Profit", "value": "Profit"},
                            ],
                            value="Sales",
                            inline=True,
                        ),
                    ],
                ),
                
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
                    children=[
                        html.Label("Segment evaluation (forecast by)"),
                        dcc.Dropdown(
                            id="forecast_groupby",
                            options=[
                                {"label": "None", "value": "None"},
                                {"label": "Category", "value": "Category"},
                                {"label": "Region", "value": "Region"},
                                {"label": "Segment", "value": "Segment"},
                            ],
                            value="None",
                            clearable=False,
                        ),
                    ],
                ),
            ],
        ),

        # -------- KPI Row --------
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(11, 1fr)",
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

                html.Div(id="kpi_forecast_6m", style=KPI_CARD_STYLE),
                html.Div(id="kpi_growth_6m", style=KPI_CARD_STYLE),
                html.Div(id="kpi_rmse", style=KPI_CARD_STYLE),
                html.Div(id="kpi_last6_actual", style=KPI_CARD_STYLE),
                html.Div(id="kpi_forecast_delta", style=KPI_CARD_STYLE),
            ],
        ),

        # -------- Charts --------
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                dcc.Graph(id="sales_trend"),
                dcc.Graph(id="profit_trend"),
            ],
        ),

        html.Div(style={"height": "8px"}),

        dcc.Graph(id="forecast_chart"),

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
# -------- Business Insights --------
html.H3("Business Insights (filtered view)"),
dcc.Markdown(id="business_insights", style={"backgroundColor": "#fafafa", "border": "1px solid #eee", "borderRadius": "10px", "padding": "10px"}),

html.Div(style={"height": "8px"}),

# -------- Customer Segmentation (RFM) --------
html.H3("Customer Segmentation (RFM)"),
html.Div(
    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
    children=[
        dcc.Graph(id="rfm_cluster_chart"),
        dash_table.DataTable(
            id="rfm_table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "6px"},
        ),
    ],
),
html.Div(id="rfm_note", style={"marginTop": "6px", "color": "#666", "fontSize": 12}),

html.Div(style={"height": "12px"}),

        # -------- Segment evaluation table --------
        html.H3("Segment evaluation (forecast KPIs)"),
        dash_table.DataTable(
            id="segment_table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "6px"},
        ),

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

    Output("kpi_forecast_6m", "children"),
    Output("kpi_growth_6m", "children"),
    Output("kpi_rmse", "children"),
    Output("kpi_last6_actual", "children"),
    Output("kpi_forecast_delta", "children"),

    Output("sales_trend", "figure"),
    Output("profit_trend", "figure"),
    Output("forecast_chart", "figure"),
    Output("profit_by_category", "figure"),
    Output("sales_by_category", "figure"),
    Output("discount_vs_profit", "figure"),

    Output("segment_table", "data"),
    Output("segment_table", "columns"),

    Output("data_table", "data"),
    Output("data_table", "columns"),

    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("category_dd", "value"),
    Input("region_dd", "value"),
    Input("segment_dd", "value"),
    Input("forecast_metric", "value"),
    Input("forecast_groupby", "value"),
)
def update_dashboard(start_date, end_date, categories, regions, segments, forecast_metric_value, forecast_groupby_value):
    # -----------------------------
    # Filter data first
    # -----------------------------
    filtered = df.copy()

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

    # -----------------------------
    # Core KPIs
    # -----------------------------
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

    # -----------------------------
    # Trends (monthly)
    # -----------------------------
    trend = (
        filtered.set_index("Order Date")[["Sales", "Profit"]]
        .resample("ME")
        .sum()
        .reset_index()
    )
    fig_sales = px.line(trend, x="Order Date", y="Sales", title="Monthly Sales Trend")
    fig_profit = px.line(trend, x="Order Date", y="Profit", title="Monthly Profit Trend")

    # -----------------------------
    # Category charts
    # -----------------------------
    cat_perf = (
        filtered.groupby("Category", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values("Profit", ascending=False)
    )
    fig_cat_profit = px.bar(cat_perf, x="Category", y="Profit", title="Profit by Category")
    fig_cat_sales = px.bar(cat_perf, x="Category", y="Sales", title="Sales by Category")

    # -----------------------------
    # Discount vs Profit (sample)
    # -----------------------------
    scatter_sample = filtered[["Discount", "Profit"]].dropna()
    if len(scatter_sample) > 3000:
        scatter_sample = scatter_sample.sample(3000, random_state=42)

    fig_scatter = px.scatter(
        scatter_sample, x="Discount", y="Profit",
        title="Discount vs Profit (Sampled)", opacity=0.5
    )

    # -----------------------------
    # Forecast (Sales/Profit toggle)
    # -----------------------------
    metric_col = "Sales" if forecast_metric_value == "Sales" else "Profit"
    hist, fc, lower, upper, metrics = forecast_metric(filtered, value_col=metric_col, periods=6)

    # >>> THIS is where your delta code belongs <<<
    last6_actual_sum = metrics.get("last6_actual_sum", np.nan)
    forecast_sum = metrics.get("forecast_sum", np.nan)
    delta = forecast_sum - last6_actual_sum


    growth_pct = metrics.get("growth_pct", np.nan)

    # Holdout validation (last 12 months) — used for RMSE/MAPE displayed on dashboard
    hv = holdout_validation(filtered, value_col=metric_col, test_months=12)
    rmse = hv.get("rmse", np.nan)
    mape = hv.get("mape", np.nan)

    # Fallback to seasonal backtest metrics if holdout not available
    if pd.isna(rmse):
        rmse = metrics.get("rmse", np.nan)
    if pd.isna(mape):
        mape = metrics.get("mape", np.nan)

    kpi_forecast_6m = _kpi_block("Forecast (Next 6M)", _format_int(forecast_sum))
    kpi_growth_6m = _kpi_block("Forecast Growth", _format_pct(growth_pct))
    kpi_rmse = _kpi_block("RMSE (Holdout 12M)", _format_int(rmse))
    kpi_last6_actual = _kpi_block("Actual (Last 6M)", _format_int(last6_actual_sum))
    kpi_forecast_delta = _kpi_block("Forecast Δ (6M)", _format_int(delta))

    fig_forecast = go.Figure()

    # CI band
    fig_forecast.add_trace(go.Scatter(
        x=fc.index, y=upper.values,
        mode="lines", line=dict(width=0),
        showlegend=False, name="Upper"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=fc.index, y=lower.values,
        mode="lines", fill="tonexty",
        line=dict(width=0),
        name="95% CI"
    ))

    # Actual + Forecast
    fig_forecast.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name=f"Actual {metric_col}"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=fc.index, y=fc.values,
        mode="lines", name="Forecast",
        line=dict(dash="dash")
    ))

    title_metric = "MAPE" if not pd.isna(mape) else "MAPE"
    title_val = f"{mape:.2f}%" if not pd.isna(mape) else "N/A"
    fig_forecast.update_layout(
        title=f"{metric_col} Forecast (Next 6 Months) | {title_metric}: {title_val}"
    )

    # -----------------------------
    
    # -----------------------------
    # Segment evaluation table
    # -----------------------------
    seg_data = []
    seg_columns = []

    if forecast_groupby_value and forecast_groupby_value != "None":
        seg_df = forecast_segment_metrics(
            filtered,
            group_col=forecast_groupby_value,
            value_col=metric_col,
            periods=6,
            min_months=6,
        )

        if not seg_df.empty:
            # nicer formatting
            show = seg_df.copy()
            for c in ["forecast_sum", "last6_actual_sum", "growth_pct", "mape", "rmse"]:
                if c in show.columns:
                    show[c] = show[c].round(2)

            seg_data = show.to_dict("records")
            seg_columns = [{"name": c, "id": c} for c in show.columns]
        else:
            seg_data = [{"Note": "No segment KPI rows produced. Try broadening filters or choose a different grouping."}]
            seg_columns = [{"name": "Note", "id": "Note"}]
    # Data table preview
    # -----------------------------
    preview = filtered.head(200).copy()
    table_data = preview.to_dict("records")                  # MUST be list[dict]
    table_columns = [{"name": c, "id": c} for c in preview.columns]  # MUST be list[dict]

    # -----------------------------
    # Return (MUST match Outputs order)
    # -----------------------------
    return (
        kpi_sales,
        kpi_profit,
        kpi_margin,
        kpi_discount,
        kpi_orders,
        kpi_customers,

        kpi_forecast_6m,
        kpi_growth_6m,
        kpi_rmse,
        kpi_last6_actual,
        kpi_forecast_delta,

        fig_sales,
        fig_profit,
        fig_forecast,
        fig_cat_profit,
        fig_cat_sales,
        fig_scatter,

        seg_data,
        seg_columns,

        table_data,
        table_columns,
    )


@app.callback(
    Output("business_insights", "children"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("category_dd", "value"),
    Input("region_dd", "value"),
    Input("segment_dd", "value"),
)
def update_business_insights(start_date, end_date, categories, regions, segments):
    f = df.copy()
    if start_date:
        f = f[f["Order Date"] >= pd.to_datetime(start_date)]
    if end_date:
        f = f[f["Order Date"] <= pd.to_datetime(end_date)]
    if categories:
        f = f[f["Category"].isin(categories)]
    if regions:
        f = f[f["Region"].isin(regions)]
    if segments:
        f = f[f["Segment"].isin(segments)]

    if f.empty:
        return "No data available for current filters."

    # Basic insights (lean, course-project friendly)
    total_sales = f["Sales"].sum()
    total_profit = f["Profit"].sum()
    margin = (total_profit / total_sales * 100.0) if total_sales else np.nan

    # top category by sales and profit
    cat = f.groupby("Category", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    top_sales_cat = cat.sort_values("Sales", ascending=False).iloc[0]["Category"]
    top_profit_cat = cat.sort_values("Profit", ascending=False).iloc[0]["Category"]

    reg = f.groupby("Region", as_index=False).agg(Profit=("Profit", "sum")).sort_values("Profit", ascending=False)
    top_region = reg.iloc[0]["Region"]

    # discount effect quick indicator
    corr = f[["Discount", "Profit"]].dropna().corr(numeric_only=True).iloc[0, 1]
    corr_txt = "negative" if corr < -0.05 else ("positive" if corr > 0.05 else "weak/none")

    bullets = [
        f"**Total Sales:** {total_sales:,.0f} | **Total Profit:** {total_profit:,.0f} | **Margin:** {margin:.2f}%",
        f"**Top Category (Sales):** {top_sales_cat} | **Top Category (Profit):** {top_profit_cat}",
        f"**Most Profitable Region:** {top_region}",
        f"**Discount vs Profit relationship:** {corr_txt} (corr = {corr:.2f})",
        "Use these insights to prioritize profitable categories/regions and monitor discounting impacts.",
    ]

    return "\n".join([f"- {b}" for b in bullets])


@app.callback(
    Output("rfm_cluster_chart", "figure"),
    Output("rfm_table", "data"),
    Output("rfm_table", "columns"),
    Output("rfm_note", "children"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
    Input("category_dd", "value"),
    Input("region_dd", "value"),
    Input("segment_dd", "value"),
)
def update_rfm(start_date, end_date, categories, regions, segments):
    f = df.copy()
    if start_date:
        f = f[f["Order Date"] >= pd.to_datetime(start_date)]
    if end_date:
        f = f[f["Order Date"] <= pd.to_datetime(end_date)]
    if categories:
        f = f[f["Category"].isin(categories)]
    if regions:
        f = f[f["Region"].isin(regions)]
    if segments:
        f = f[f["Segment"].isin(segments)]

    empty_fig = go.Figure().update_layout(
        title="RFM Clusters (k=4)",
        xaxis_title="Cluster",
        yaxis_title="Customers"
    )

    # Need enough customers for clustering to be meaningful
    n_customers = f["Customer ID"].nunique()
    if n_customers < 20:
        note = f"RFM requires enough customers. Current filtered customers: {n_customers}. Broaden filters to see clusters."
        return empty_fig, [], [], note

    rfm = compute_rfm(f)
    rfm = run_kmeans(rfm, k=4)
    prof = profile_clusters(rfm)
    prof = label_clusters(prof)

    # Cluster size chart
    counts = prof[["Customers"]].reset_index().rename(columns={"index": "Cluster"})
    fig = px.bar(counts, x="Cluster", y="Customers", title="RFM Cluster Sizes (k=4)")

    # Table data
    show = prof.reset_index().rename(columns={"index": "Cluster"})
    data = show.to_dict("records")
    columns = [{"name": c, "id": c} for c in show.columns]

    note = "Interpretation: lower Recency + higher Frequency/Monetary typically indicates higher-value customers (e.g., Champions)."
    return fig, data, columns, note

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
