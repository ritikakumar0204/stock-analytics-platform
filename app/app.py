import json
import os
import textwrap
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud import bigquery

from forecast_page import render_forecast


st.set_page_config(page_title="Stock Forecasting Engine", layout="wide")


# -------- Sidebar Navigation (buttons by section) --------
st.sidebar.title("📂 Menu")

nav_sections = [
    ("Home", [("Home", "📈")]),
    ("Data Ingestion", [("Ingestion", "⚙️")]),
    ("Analytics", [("EDA", "📊"), ("Reports", "📑"), ("Ad Hoc Report", "🧾")]),
    ("Forecast", [("Forecast", "🔮")]),
]

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Home"

for idx, (section, items) in enumerate(nav_sections):
    st.sidebar.markdown(f"**{section}**")
    for name, icon in items:
        label = f"{icon} {name}"
        if st.sidebar.button(label, key=f"nav_{name}"):
            st.session_state.selected_tab = name
    if idx < len(nav_sections) - 1:
        st.sidebar.markdown("---")

page = st.session_state.selected_tab


# -------- Shared config --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
PRICES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.prices" if PROJECT_ID and DATASET_ID else None
HISTORICAL_CSV = PROJECT_ROOT / "dataset" / "historical_stocks_clean.csv"


@st.cache_resource(show_spinner=False)
def get_bq_client():
    if not SERVICE_ACCOUNT_KEY:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_KEY env var (path to service account JSON).")

    key_path = Path(SERVICE_ACCOUNT_KEY)
    if not key_path.is_absolute():
        candidate = PROJECT_ROOT / key_path
        key_path = candidate if candidate.exists() else key_path

    if not key_path.exists():
        raise RuntimeError(f"Service account key not found at: {key_path}")

    return bigquery.Client.from_service_account_json(key_path)


@st.cache_data(show_spinner=False)
def fetch_ticker_data(ticker: str) -> pd.DataFrame:
    if not PRICES_TABLE:
        raise RuntimeError("GCP_PROJECT_ID or GCP_DATASET_ID is not configured.")

    client = get_bq_client()
    query = f"""
        SELECT *
        FROM `{PRICES_TABLE}`
        WHERE ticker = @ticker
        ORDER BY date DESC
        LIMIT 500
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


@st.cache_data(show_spinner=False)
def run_ad_hoc_query(sql: str) -> pd.DataFrame:
    client = get_bq_client()
    job = client.query(sql)
    return job.result().to_dataframe()


@st.cache_data(show_spinner=False)
def load_metadata() -> list[dict]:
    meta_path = PROJECT_ROOT / "dataset" / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found at {meta_path}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    cleaned = []
    for entry in raw:
        industry = entry.get("industry", "")
        industry = industry.replace("ƒ?", " ")
        cleaned.append({**entry, "industry": industry})
    return cleaned


@st.cache_data(show_spinner=False)
def load_historical_data() -> pd.DataFrame:
    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(f"Historical CSV not found at {HISTORICAL_CSV}")

    df = pd.read_csv(HISTORICAL_CSV)
    df = df.rename(columns=str.title)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    current_year = pd.Timestamp.now().year
    df = df[df["Date"].dt.year == current_year]
    df = df.sort_values(["Ticker", "Date"])

    df["MA50"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(window=50, min_periods=1).mean())
    df["MA70"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(window=70, min_periods=1).mean())
    df["DailyReturnPct"] = df.groupby("Ticker")["Close"].transform(lambda s: s.pct_change(fill_method=None) * 100)
    return df


def render_report_visual(stem: str, df: pd.DataFrame):
    """Render visualization for a report; currently only used for ticker correlation."""
    if stem == "ticker_correlation" and df.shape[1] > 1:
        df_corr = df.copy()
        if {"ticker_a", "ticker_b", "correlation"} <= set(df_corr.columns):
            df_corr["correlation"] = pd.to_numeric(df_corr["correlation"], errors="coerce")
            tickers = sorted(set(df_corr["ticker_a"]).union(df_corr["ticker_b"]))
            matrix = pd.DataFrame(1.0, index=tickers, columns=tickers)
            for _, row in df_corr.dropna(subset=["correlation"]).iterrows():
                a, b, c = row["ticker_a"], row["ticker_b"], row["correlation"]
                if a in matrix.index and b in matrix.columns:
                    matrix.loc[a, b] = c
                    matrix.loc[b, a] = c
            heatmap_data = matrix.reset_index().melt(id_vars="index")
            heatmap_data.columns = ["TickerA", "TickerB", "Correlation"]
        else:
            if "ticker" in df_corr.columns:
                df_corr = df_corr.set_index("ticker")
            df_corr = df_corr.select_dtypes(include=["number"])
            if df_corr.empty:
                return
            matrix = df_corr.corr()
            heatmap_data = matrix.reset_index().melt(id_vars="index")
            heatmap_data.columns = ["TickerA", "TickerB", "Correlation"]

        chart = (
            alt.Chart(heatmap_data)
            .mark_rect()
            .encode(
                x=alt.X("TickerA:N", title=""),
                y=alt.Y("TickerB:N", title=""),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["TickerA", "TickerB", "Correlation"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        return
    # For other reports, no visualization is rendered.


# -------- HOME --------
if page == "Home":
    st.title("📈 Stock market forecasting engine")
    st.caption("Navigate via the sidebar to ingest data, explore analytics, or run forecasts.")

    try:
        companies = load_metadata()
    except Exception as exc:
        st.error(f"Failed to load metadata: {exc}")
        companies = []

    if companies:
        st.subheader("Featured tickers")
        cols = st.columns(2, gap="medium")
        for idx, entry in enumerate(companies):
            with cols[idx % 2]:
                ticker = entry.get("ticker", "")
                name = entry.get("company_name", "")
                sector = entry.get("sector", "")
                industry = entry.get("industry", "")
                exch = entry.get("exchange", "")
                mcap = entry.get("market_cap", None)
                if mcap:
                    if mcap >= 1e12:
                        mcap_str = f"${mcap/1e12:,.2f}T"
                    else:
                        mcap_str = f"${mcap/1e9:,.1f}B"
                else:
                    mcap_str = "N/A"

                card_html = textwrap.dedent(
                    f"""
                    <div style="
                        border: 1px solid #2d3748;
                        border-radius: 10px;
                        padding: 14px;
                        background: #1e293b;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
                        margin-bottom: 12px;
                    ">
                        <div style="font-size: 22px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">
                            {ticker}
                        </div>
                        <div style="font-size: 14px; color:#cbd5e1; margin-bottom: 6px;">
                            {name}
                        </div>
                        <div style="font-size: 12px; color:#94a3b8;">
                            <span style="color:#38bdf8;">{sector}</span> · {industry}
                        </div>
                        <div style="font-size: 12px; color:#94a3b8; margin-top: 4px;">
                            Exchange: {exch}
                        </div>
                        <div style="
                            font-size: 14px;
                            font-weight: 600;
                            color: #f8fafc;
                            background: rgba(59,130,246,0.18);
                            padding: 4px 8px;
                            border-radius: 6px;
                            width: fit-content;
                            margin-top: 10px;
                        ">
                            Market Cap: {mcap_str}
                        </div>
                    </div>
                    """
                )
                st.markdown(card_html, unsafe_allow_html=True)




# -------- INGESTION --------
elif page == "Ingestion":
    st.title("⚙️ Data Ingestion")
    available_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL"]
    ticker = st.selectbox("Choose a ticker:", available_tickers, index=0)
    creds_ready = PROJECT_ID and DATASET_ID and SERVICE_ACCOUNT_KEY

    if not creds_ready:
        st.warning("Configure GCP_PROJECT_ID, GCP_DATASET_ID, and GCP_SERVICE_ACCOUNT_KEY to enable BigQuery fetches.")

    if st.button("Fetch Data", disabled=not creds_ready):
        with st.spinner(f"Fetching data for {ticker} from BigQuery..."):
            try:
                df = fetch_ticker_data(ticker)
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")
            else:
                if df.empty:
                    st.info(f"No rows returned for {ticker}.")
                else:
                    st.success(f"Fetched {len(df)} rows for {ticker}. Showing up to 500 latest.")
                    if "date" in df.columns and "close" in df.columns:
                        df_sorted = df.copy()
                        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
                        df_sorted = df_sorted.sort_values("date")
                        st.line_chart(df_sorted.set_index("date")["close"])
                    st.dataframe(df)


# -------- EDA --------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    try:
        hist_df = load_historical_data()
    except Exception as exc:
        st.error(f"Failed to load historical data: {exc}")
        st.stop()

    tickers = sorted(hist_df["Ticker"].unique())
    selected = st.multiselect("Choose tickers:", tickers, default=tickers)

    filtered = hist_df[hist_df["Ticker"].isin(selected)]
    if filtered.empty:
        st.info("Select at least one ticker to view charts.")
    else:
        st.subheader("Close Price Comparison")
        pivot_close = filtered.pivot_table(index="Date", columns="Ticker", values="Close")
        st.line_chart(pivot_close)

        st.subheader("Average Volume by Ticker")
        vol_df = filtered.groupby("Ticker")["Volume"].mean().reset_index(name="AvgVolume")
        st.bar_chart(vol_df.set_index("Ticker"))

        st.subheader("Daily Return (%) Trend")
        pivot_ret = filtered.pivot_table(index="Date", columns="Ticker", values="DailyReturnPct")
        st.line_chart(pivot_ret)

        st.subheader("Distribution of Daily Returns (%)")
        ret_series = filtered["DailyReturnPct"].dropna()
        if ret_series.empty:
            st.info("No daily return data available for the selected tickers.")
        else:
            ret_df = pd.DataFrame({"DailyReturnPct": ret_series})
            hist_chart = (
                alt.Chart(ret_df)
                .mark_bar(color="#7BA7E0", opacity=0.75)
                .encode(
                    x=alt.X("DailyReturnPct:Q", bin=alt.Bin(maxbins=50), title="Daily Return (%)"),
                    y=alt.Y("count()", title="Frequency"),
                    tooltip=[alt.Tooltip("count()", title="Frequency")],
                )
            )
            density_chart = (
                alt.Chart(ret_df)
                .transform_density(
                    density="DailyReturnPct",
                    as_=["DailyReturnPct", "density"],
                    bandwidth=1.8,
                )
                .transform_calculate(DailyReturnPctShift="datum.DailyReturnPct + 0.05")
                .mark_area(color="#4C78A8", opacity=0.35)
                .encode(
                    x=alt.X("DailyReturnPctShift:Q", title="Daily Return (%)"),
                    y=alt.Y("density:Q", title="Density", axis=alt.Axis(orient="right")),
                )
            )
            combined_chart = (hist_chart + density_chart).resolve_scale(y="independent")
            st.altair_chart(combined_chart, use_container_width=True)

        st.caption("Volatility (std dev of daily returns, %)")
        vol_summary = (
            filtered.groupby("Ticker")["DailyReturnPct"]
            .std()
            .sort_values(ascending=False)
            .reset_index(name="StdDevPct")
        )
        st.dataframe(vol_summary)


# -------- REPORTS --------
elif page == "Reports":
    st.title("📑 SQL Reports")
    st.caption("Results from precomputed CSVs with the option to view the SQL used.")

    results_dir = PROJECT_ROOT / "sql" / "results"
    queries_dir = PROJECT_ROOT / "sql" / "queries"

    report_files = [
        ("Average Volatility", "average_volatility"),
        ("Daily Return Rank", "daily_return_rank"),
        ("MA Crossover", "ma_crossover"),
        ("Pct Change Monthly", "pct_change_monthly"),
        ("Rolling Drawdown", "rolling_drawdown"),
        ("Ticker Correlation", "ticker_correlation"),
        ("Top Stocks 30 Days", "top_stocks_30_days"),
        ("Volatility Shock", "volatility_shock"),
    ]

    report_descriptions = {
        "Average Volatility": "Average short- and medium-term volatility per ticker to spot consistently volatile names.",
        "Daily Return Rank": "Ranks tickers by daily returns to highlight recent outperformers.",
        "MA Crossover": "Tracks moving-average crossovers that may indicate trend shifts.",
        "Pct Change Monthly": "Month-over-month percentage changes by ticker to show recent momentum.",
        "Rolling Drawdown": "Rolling peak-to-trough declines to understand downside risk over time.",
        "Ticker Correlation": "Pairwise correlations across tickers to identify co-movement.",
        "Top Stocks 30 Days": "Top tickers over the last 30 days based on performance metrics.",
        "Volatility Shock": "Flags volatility spikes versus baseline to surface unusual moves.",
    }

    report_map = {title: stem for title, stem in report_files}
    selected_title = st.selectbox("Choose a report to view:", list(report_map.keys()))

    stem = report_map[selected_title]
    csv_path = results_dir / f"{stem}.csv"
    sql_path = queries_dir / f"{stem}.sql"

    st.subheader(selected_title)
    if selected_title in report_descriptions:
        st.caption(report_descriptions[selected_title])
    try:
        df = pd.read_csv(csv_path)
        render_report_visual(stem, df)
        st.dataframe(df, use_container_width=True)
    except FileNotFoundError:
        st.error(f"Missing CSV file: {csv_path}")

    if sql_path.exists():
        with st.expander("View query"):
            st.code(sql_path.read_text(), language="sql")
    else:
        st.warning(f"Query file not found: {sql_path}")


# -------- AD HOC REPORT --------
elif page == "Ad Hoc Report":
    st.title("🧾 Ad Hoc Report")
    st.caption("Run read-only SELECT queries against your BigQuery prices table.")

    creds_ready = PROJECT_ID and DATASET_ID and SERVICE_ACCOUNT_KEY
    if not creds_ready:
        st.warning("Configure GCP_PROJECT_ID, GCP_DATASET_ID, and GCP_SERVICE_ACCOUNT_KEY to enable ad hoc queries.")
        st.stop()

    default_table = PRICES_TABLE if PRICES_TABLE else "project.dataset.prices"
    default_sql = f"""SELECT
  ticker,
  AVG(close) AS avg_close,
  DATE(date) AS date
FROM `{default_table}`
WHERE ticker = 'AAPL'
GROUP BY ticker, date
ORDER BY date DESC
LIMIT 10"""

    sql_text = st.text_area("SQL (SELECT only)", height=220)
    if st.button("Run query", type="primary"):
        if not sql_text.strip():
            st.error("Enter a SQL query to run.")
        elif not sql_text.strip().lower().startswith("select"):
            st.error("Only SELECT queries are allowed in this console.")
        else:
            with st.spinner("Running query..."):
                try:
                    df = run_ad_hoc_query(sql_text)
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                else:
                    if df.empty:
                        st.info("Query returned no rows.")
                    else:
                        st.success(f"Returned {len(df)} rows.")
                        st.dataframe(df, use_container_width=True)


# -------- FORECAST --------
elif page == "Forecast":
    render_forecast()
