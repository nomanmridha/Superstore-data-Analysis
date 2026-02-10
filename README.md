# 📊 Superstore Analytics Dashboard  

### Business Intelligence + Forecasting + Customer Segmentation (RFM)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Dash](https://img.shields.io/badge/Dash-Plotly-black)
![Pandas](https://img.shields.io/badge/Pandas-Analytics-purple)
![KMeans](https://img.shields.io/badge/KMeans-Clustering-green)
![Data Analysis](https://img.shields.io/badge/Data-Analysis-00B2A9?style=for-the-badge&logo=chart-line&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Excel](https://img.shields.io/badge/Excel-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🎯 Project Overview

This project transforms the **Superstore retail dataset** into a complete **Business Intelligence and Advanced Analytics Dashboard**.
**Superstore Sales Analytics** is an end-to-end Python project that analyzes a large retail dataset to uncover business insights, forecast demand, and segment customers for targeted strategy.  
This project was completed as part of the **Advanced Analytics** course and delivers an interactive **Dash dashboard** that enables stakeholders to explore trends, evaluate forecast performance, and profile customer segments.

The objective was to:

- Build a dynamic decision-support dashboard  
- Forecast future sales and validate model performance  
- Evaluate forecast KPIs across business segments  
- Segment customers using RFM + clustering  
- Generate actionable business insights

---

## 🧠 Problem Statement

Retail managers often need actionable insights from their sales data to:
- Understand historical performance by product, region, and segment
- Forecast future demand with performance validation
- Identify high-value customer segments
- Inform price, inventory, and marketing decisions

This project provides a **data pipeline, analytics models, and a user-friendly dashboard** to support those needs.

---

# 📌 Data Source
 
Kaggle Link:  👉 https://www.kaggle.com/datasets/vivek468/superstore-dataset-final  

---

## 🧰 Tech Stack

| Feature | Technology |
|----------|------------|
| Data Cleaning | Pandas, NumPy |
| EDA | Jupyter Notebook |
| Forecasting | ARIMA-style + Holdout Validation |
| Prototype Forecast | Prophet |
| Customer Segmentation | RFM + KMeans |
| Visualization | Plotly |
| Dashboard | Dash |
| Environment | Python 3.12 (venv312) |
| Version Control | Git + GitHub |

---

## 🚀 Project Journey (How It Evolved)
This project evolved through structured analytical stages:

## Phase 1 — Raw Data Exploration

We began with loading and inspecting the raw dataset.

![Raw Dataset Preview](assets/figures/Raw_dataset.png)

Notebook:  
`notebooks/01_data_loading_cleaning.ipynb`  👉 **[Data cleaning](notebooks/01_data_loading_cleaning.ipynb)**

Tasks performed:
- Schema inspection  
- Data type corrections  
- Missing value validation  
- Duplicate checks  
- Date parsing  

---

## Phase 2 — Data Cleaning & Processed Dataset

After cleaning and transformation:

![Cleaned Dataset Preview](assets/figures/Cleaned_Dataset.png)

The raw dataset underwent:
- Datatype conversion (dates)
- Duplicate removal
- Missing value checks
- Profit scaling correction (final validation)
- Aggregation consistency
- Sales & Profit Totals
- Processed dataset exported as `superstore_processed.csv`  👉 **[Processed Data](data/processed)**

---

## Phase 3 — Exploratory Data Analysis (EDA)

Notebook:  
`notebooks/02_eda.ipynb`  👉 **[Exploratory data analysis](notebooks/02_eda.ipynb)**

### Monthly Sales Trend (2014–2017)

![Monthly Sales Trend](assets/figures/Monthly_Sales_Trend_(2014-2017).png)

### Monthly Profit Trend (2014–2017)

![Monthly Profit Trend](assets/figures/Monthly_Profit_Trend_(2014-2017).png)

### Sales vs Profit Over Time

![Sales vs Profit](assets/figures/Sales_vs_Profit_Over_time_(2014-2017).png)

### Discount vs Profit Relationship

![Discount vs Profit](assets/figures/Discount_vs_Profit.png)

Key findings:
- Discount negatively impacts profit  
- Clear seasonal sales patterns  
- Performance differences across categories  

---

# 🧠 Customer Segmentation (RFM + Clustering)

Notebook:  
`notebooks/03_rfm_analysis.ipynb`  👉 **[RFM + Clustering](notebooks/03_rfm_analysis.ipynb)**

RFM Metrics:
- **Recency** – Days since last purchase  
- **Frequency** – Number of orders  
- **Monetary** – Total purchase value  

### RFM Visualizations

![Recency vs Frequency](assets/figures/Customer_Segments_Recency_vs_Frequency.png)
![Frequency vs Monetary](assets/figures/Customer_Segments_Frequency_vs_Monetary.png)
![Discount vs Profit - Customers](assets/figures/Customer_Segments_Discount_vs_Profit.png)

### Optimal K Determination

![Elbow Method](assets/figures/Elbow_Method_for_Optimal_K.png)

Final model: **KMeans (k = 4)**

Segments identified:
- Champions  
- Loyal Customers  
- Big Spenders  
- Lost / At Risk  

---

# 📈 Forecasting & Validation

Notebook:  
`notebooks/04_forecasting.ipynb`  👉 **[Forecasting](notebooks/04_forecasting.ipynb)**

Methods explored:
- ARIMA-style modeling  
- Prophet prototype  

![Prophet Forecast](assets/figures/Sales_forecast_using_prophet.png)

Final dashboard implementation includes:
- 6-month forecast  
- 12-month holdout validation  
- RMSE & MAPE metrics  

### Sales Forecast with MAPE

![Sales Forecast](assets/figures/Sales_Forecast_next_6_month_with_MAPE.png)

### MAPE Metric

![MAPE](assets/figures/mape.png)

---

# 🖥️ Dashboard Evolution

Initially developed:
- `streamlit_app.py`  👉 **[Streamlit](app/streamlit_app.py)**
- `test_app.py`       👉 **[Streamlit Test](app/test_app.py)**

Due to frontend layout and rendering constraints, migrated to **Dash**.

Final architecture:
- `rebuild_processed.py`  👉 **[Rebuilding](app/rebuild_processed.py)**
- `forecasting.py`        👉 **[Forecasting](app/forecasting.py)**
- `rfm_model.py`          👉 **[RFM Model](app/rfm_model.py)**
- `dash_app.py`           👉 **[Dash Dashboard](app/dash_app.py)**

---

# 📊 Final Dashboard Preview

![Dashboard Final Preview](assets/figures/Dashboard_final_Preview.png)

---

### **Key Business KPIs**

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
  
---

### **Time Series Trend Analysis Dashboard**

View:
- Monthly Sales Trend
- Monthly Profit Trend
  
![Monthly Sales Dashboard](assets/figures/Monthly_Sales_Trend_Dashboard.png)
![Monthly Profit Dashboard](assets/figures/Monthly_Profit_Trend_Dashboard.png)

---

## Category Performance

View:
- Sales by Category
- Profit by Category

![Sales by Category](assets/figures/Sales_by_Category_Dashboard.png)
![Profit by Category](assets/figures/Profit_graph_by_category_Dashboard.png)

---

## Discount Analysis (Dashboard)

![Discount vs Profit Sampled](assets/figures/Discount_vs_profit_sampled_Dashboard_graph.png)

---

## Segment Evaluation (Forecast KPIs)

Evaluates forecast performance by:
- Category  
- Region  
- Retail Segment  

Outputs include:
- forecast_sum  
- last6_actual_sum  
- growth_pct  
- mape  
- rmse  

---

## Customer Segmentation Dashboard

![RFM Dashboard Graph](assets/figures/Customer_Segmentation_RFM_dashboard_graph.png)

Displays:
- Cluster size distribution  
- RFM profiling table  
- Segment labeling  

---

## Segment-Specific Forecast Examples

View:
- Champion Forecast
![Champion Forecast](assets/figures/Champion_Forecast.png)

View:
- Loyal Forecast
![Loyal Forecast](assets/figures/Loyal_Forecast.png)

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

## 👤 Author
**Course:** Advance Analytics (WiSe26)  
**University:** Fachhochschule Südwestfalen  
**Supervisor:** Prof. Dr. Christian Leubner  
**Project Type:** Individual Research Project

![FH Südwestfalen](https://img.shields.io/badge/FH-S%C3%BCdwestfalen-0083CC?style=for-the-badge&logo=university&logoColor=white)
![Research Project](https://img.shields.io/badge/Research-Project-6A1B9A?style=for-the-badge&logo=graduation-cap&logoColor=white)

## 🤝 Connect & Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-abdullah-al-noman-333aa4155/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nomanmridha/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:noman.hr.18@gmail.com)
* **University:** FH Südwestfalen – Advanced Analytics

---

📌 *This repository demonstrates how academic projects can be elevated to industry-ready analytics portfolios through strong documentation, business framing, and technical rigor.*

