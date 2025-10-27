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
1. Average daily volatility by ticker or sector  
2. Top-performing stocks by month or quarter  
3. Correlation between stocks and market indices  
4. Model prediction accuracy trend over time  
5. Moving-average crossover alerts  
6. Percentage price change by period  
7. Feature importance / lag effect summary  
8. ETL latency and API call success rate  


---

## 📈 Progress Tracking
Team progress and milestone completion have been documented in  
[`progress_tracker.xlsx`](https://northeastern-my.sharepoint.com/:x:/r/personal/kumar_riti_northeastern_edu/Documents/progress_tracker.xlsx?d=wa7fec4b0e5154b40be4c327f848a7cf3&csf=1&web=1&e=UwO6qY)).  
This Excel sheet logs major deliverables, deadlines, and issue resolutions for transparency.

