# 📊 Superstore Analytics Dashboard  
*Business Intelligence + Forecasting + Customer Segmentation (RFM)*

**Superstore Sales Analytics** is an end-to-end Python project that analyzes a large retail dataset to uncover business insights, forecast demand, and segment customers for targeted strategy.  
This project was completed as part of the **Advanced Analytics** course and delivers an interactive **Dash dashboard** that enables stakeholders to explore trends, evaluate forecast performance, and profile customer segments.

---

## 🧠 Problem Statement

Retail managers often need actionable insights from their sales data to:
- Understand historical performance by product, region, and segment
- Forecast future demand with performance validation
- Identify high-value customer segments
- Inform price, inventory, and marketing decisions

This project provides a **data pipeline, analytics models, and a user-friendly dashboard** to support those needs.

---

## 📁 Repository Structure

```
Superstore-data-Analysis/
├── app/
│   ├── dash_app.py
│   ├── forecasting.py
│   ├── rfm_model.py
│   ├── rebuild_processed.py
├── data/
│   ├── raw/
│   │   └── superstore_clean.csv
│   └── processed/
│       └── superstore_processed.csv
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_rfm_analysis.ipynb
│   └── 04_forecasting.ipynb
├── assets/
│   └── dashboard_screenshots/
├── slides/
├── reports/
├── README.md
└── requirements.txt
```

---

## 🧰 Tech Stack

| Feature | Tech |
|--------|------|
| Data Cleaning & EDA | Python, Pandas, NumPy |
| Visualization | Plotly, Dash |
| Forecasting | ARIMA, Holdout validation |
| Customer Segmentation | RFM, KMeans (k=4) |
| Interactive Dashboard | Dash (Plotly) |
| Packaging & Requirements | venv, requirements.txt |

---

## 🚀 Project Workflow

### 1. Dataset Source

The dataset is based on the **Superstore sales dataset** (originally from Kaggle / internal business data).  
It contains sales, profit, discount, customer, order, and segmentation information.

📌 **Raw dataset preview**:
<!-- asset: raw data preview image -->
![Raw dataset preview](assets/dashboard_screenshots/raw_dataset_preview.png)

---

### 2. Data Cleaning & Preparation

The raw dataset underwent:
- Datatype conversion (dates)
- Duplicate removal
- Missing value checks
- Profit scaling correction (final validation)
- Processed dataset exported as `superstore_processed.csv`

📌 **Cleaned dataset preview**:
<!-- asset: processed data preview image -->
![Processed dataset preview](assets/dashboard_screenshots/processed_dataset_preview.png)

Notebooks:
- `01_data_loading_cleaning.ipynb`
- `02_eda.ipynb`

---

## 📈 Interactive Dashboard Features

The core product of this project is an **interactive dashboard** built with Dash that includes:

---

### 1) **Key Business KPIs**

Displays:
- Total Sales
- Total Profit
- Profit Margin
- Orders
- Customers
- Forecast (Next 6 Months)
- Forecast Growth %
- RMSE (Holdout 12M)
- Actual vs Forecast delta

📌 **Dashboard top KPIs**:
<!-- asset: dashboard_kpis.png -->
![Dashboard KPIs](assets/dashboard_screenshots/dashboard_kpis.png)

---

### 2) **Time Series Trend Analysis**

View:
- Monthly Sales Trend
- Monthly Profit Trend

📌 **Trend charts**:
<!-- asset: monthly_sales_trend.png -->
![Monthly Sales Trend](assets/dashboard_screenshots/monthly_sales_trend.png)

---

### 3) **Forecasting**

Your dashboard shows:
- **Sales Forecast** for the next 6 months
- **MAPE (%)** in the title
- Confidence intervals

📌 **Forecast preview**:
<!-- asset: sales_forecast.png -->
![Sales Forecast](assets/dashboard_screenshots/sales_forecast.png)

---

### 4) **Segment Evaluation (Forecast KPIs)**

Analyze forecast performance by:
- Category
- Region
- Retail Segment

📌 **Example Segment evaluation (Category)**:
<!-- asset: segment_eval_category.png -->
![Segment Eval - Category](assets/dashboard_screenshots/segment_eval_category.png)

---

## 🧠 Customer Segmentation (RFM + Clustering)

Using Recency-Frequency-Monetary (RFM) analysis:

1) **Recency** — Days since last purchase  
2) **Frequency** — Number of orders  
3) **Monetary** — Total sales value  

Clustering (k=4) segments customers into:
- Champions
- Loyal
- At Risk
- Big Spenders

📌 **RFM cluster sizes**:
<!-- asset: rfm_cluster_sizes.png -->
![RFM Cluster Sizes](assets/dashboard_screenshots/rfm_cluster_sizes.png)

📌 **RFM profile table**:
<!-- asset: rfm_profile_table.png -->
![RFM Profile Table](assets/dashboard_screenshots/rfm_profile_table.png)

---

## 💡 Business Insights (Auto-Generated)

The dashboard calculates insights based on filtered data:
- Top performing categories & regions
- Profit-discount relationships
- Customer segment behavior

📌 **Business insights block**:
<!-- asset: business_insights.png -->
![Business Insights](assets/dashboard_screenshots/business_insights.png)

---

## 📝 How to Run

1. Clone the repo:
```bash
git clone https://github.com/nomanmridha/Superstore-data-Analysis.git
cd Superstore-data-Analysis
```

2. Create a Python 3.12 virtual environment:
```bash
python -m venv venv312
.\venv312\Scripts\activate     # Windows
source venv312/bin/activate   # Mac/Linux
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Run the dashboard:
```bash
python app/dash_app.py
```

Visit:
```text
http://127.0.0.1:8050
```

---

## 📊 Sample Screenshots

Showcase of key outputs:

| Feature | Preview |
|--------|---------|
| Forecast Chart | ![Forecast](assets/dashboard_screenshots/sales_forecast.png) |
| Segment Eval (Region) | ![Region Eval](assets/dashboard_screenshots/segment_eval_region.png) |
| RFM Cluster | ![RFM Clusters](assets/dashboard_screenshots/rfm_cluster_sizes.png) |

---

## 📌 Limitations & Future Work

- Forecast model uses seasonal-naive method — future work could integrate SARIMA/Prophet
- RFM uses fixed k=4 — more dynamic cluster validation possible
- Additional filters (product, store) could enhance analysis

---

## 🧾 License

This project is released under the **MIT License** (see LICENSE file).

---

## 📚 Acknowledgements

- Kaggle Superstore dataset
- Plotly & Dash libraries
- Python analytics ecosystem


