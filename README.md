# stock-analytics-platform

## 📘 Project Overview
An end-to-end financial analytics and forecasting platform built using **freely available APIs** (Yahoo Finance, Alpha Vantage) and a **cloud-integrated SQL database**.  
The project demonstrates batch and streaming data ingestion, predictive modeling, and advanced SQL reporting — all wrapped into a reproducible data pipeline designed for transparency and performance.

---

## 🎯 Objectives
- Collect historical and real-time stock data using public APIs within rate-limit constraints.  
- Design a **normalized SQL schema** for storing tickers, time-series data, engineered features, and predictions.  
- Implement **batch ingestion** (historical data) and **stream ingestion** (recent updates).  
- Integrate cloud storage via **GCP BigQuery** for scalability.  
- Generate **eight advanced SQL analytical reports** measuring volatility, correlation, accuracy trends, and more.  
- Log all ETL operations for traceability and reproducibility.

---

## 👥 Team Members and Roles
| Member | Role | Key Contributions |
|:--|:--|:--|
| **Ritika Kumar** | Data Pipeline Engineer | API integration, ETL pipeline development, data cleaning, and cloud storage integration (GCP BigQuery) |
| **Raisa Vikas Furtado** | Database Architect | Database design and implementation, SQL schema creation, normalization, indexing, and ER diagram development |
| **Tarun Sethi** | ML Engineer | Feature engineering, model training and evaluation (ARIMA, Prophet, XGBoost, LSTM), and visualization dashboard design (Plotly / Tableau) |

---

## 🧰 Tools & Technologies
**Languages:** Python, SQL, Shell  
**Libraries:** Pandas, NumPy, Requests, SQLAlchemy, Scikit-learn, Matplotlib  
**Databases:** PostgreSQL / MySQL  
**Cloud:** GCP BigQuery  
**Documentation:** Overleaf (PDF report), Excel (progress tracking), GitHub (version control)

---

## 🔗 Data Sources
- [Yahoo Finance API](https://finance.yahoo.com/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/) (If required)
  
All data used is publicly available.  
No restricted, proprietary, or brokerage data has been accessed or scraped.

---

## 🗄️ Database Design
Subject to change

**Core Tables**
- `stocks` – ticker, company name, sector, industry  
- `prices` – ticker, date, open, high, low, close, volume  
- `features` – ticker, date, technical indicators (e.g., RSI, MA, volatility)  
- `predictions` – ticker, date, model name, predicted price, confidence, accuracy  

**Highlights**
- Composite keys (`ticker`, `date`) for time-series data  
- Foreign-key relationships ensuring normalization  
- Indexed columns for faster analytical queries  

---

## 📊 Analytical SQL Reports
1. Average volatility  
2. Daily return rank
3. MA crossover  
4. PCT change monthly  
5. Rolling drawdown
6. Ticker correlation 
7. Top stocks 30 days
8. Volatility shock


---
## 🚀 How to Run the Project

### **1️⃣ Install Dependencies**
Clone the repository and install all required packages:
```bash
git clone https://github.com/ritikakumar0204/stock-analytics-platform.git
cd stock-analytics-platform
pip install -r requirements.txt
```
---
### 2️⃣ Launch the Streamlit App

Run the dashboard locally:

```bash
streamlit run app.py
```

----

## 📈 Progress Tracking
Team progress and milestone completion have been documented in  
[`progress_tracker.xlsx`](https://northeastern-my.sharepoint.com/:x:/r/personal/kumar_riti_northeastern_edu/Documents/progress_tracker.xlsx?d=wa7fec4b0e5154b40be4c327f848a7cf3&csf=1&web=1&e=UwO6qY).  
This Excel sheet logs major deliverables, deadlines, and issue resolutions for transparency.

