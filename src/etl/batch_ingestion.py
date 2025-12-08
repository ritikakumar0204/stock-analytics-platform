import yfinance as yf
import pandas as pd
import datetime as dt
import logging
from google.cloud import bigquery
from dotenv import load_dotenv
import os

# -----------------------------------
# Logging
# -----------------------------------
logging.basicConfig(
    filename="batch_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------
# ENV + BigQuery Client
# -----------------------------------
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.prices"

client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)

# -----------------------------------
# Configurable parameters
# -----------------------------------
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN"]
START_DATE = "2020-01-01"
END_DATE = dt.date.today().isoformat()

# -----------------------------------
# Fetch Raw Multi-Column Data
# -----------------------------------

import time

def fetch_single_ticker(ticker, retries=1, delay=2):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE)
            if not df.empty:
                logging.info(f"{ticker}: {len(df)} rows fetched.")
                df["Ticker"] = ticker
                return df
            else:
                logging.warning(f"{ticker}: empty dataframe")
        except Exception as e:
            logging.error(f"{ticker}: attempt {attempt+1} failed: {e}")
        time.sleep(delay)

    logging.error(f"{ticker}: FAILED after {retries} retries")
    return None

def fetch_raw_data():
    logging.info("Fetching data ticker-by-ticker...")
    frames = []
    for t in TICKERS:
        df = fetch_single_ticker(t)
        if df is not None:
            frames.append(df)
    if not frames:
        raise ValueError("No tickers fetched successfully.")
    return pd.concat(frames, ignore_index=False)


# -----------------------------------
# Clean using your EXACT transformation logic
# -----------------------------------
def clean_yahoo_data(df):
    logging.info("Cleaning flat Yahoo Finance data...")

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Ticker": "ticker"
    })

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["created_at"] = pd.Timestamp.utcnow()

    df = df[["date", "open", "high", "low", "close", "volume", "ticker", "created_at"]]

    logging.info(f"Final cleaned rows: {len(df)}")
    return df

# -----------------------------------
# BigQuery Upload
# -----------------------------------
def upload_to_bigquery(df):
    try:
        job = client.load_table_from_dataframe(df, TABLE_ID)
        job.result()
        logging.info(f"Uploaded {len(df)} rows to {TABLE_ID}")
        print(f"✔ Uploaded {len(df)} rows to BigQuery.")
    except Exception as e:
        logging.error(f"Upload to BigQuery failed: {e}")
        print("❌ Upload failed. Check logs.")
        raise e

# -----------------------------------
# Batch ETL Pipeline
# -----------------------------------
def main():
    logging.info("=== BATCH INGESTION STARTED ===")

    raw = fetch_raw_data()
    clean_df = clean_yahoo_data(raw)

    # Save local CSV if needed
    clean_df.to_csv("historical_stocks_clean.csv", index=False)
    logging.info("Saved cleaned CSV locally.")

    upload_to_bigquery(clean_df)

    logging.info("=== BATCH INGESTION COMPLETED ===")
    print("✔ Batch ingestion complete.")

if __name__ == "__main__":
    main()
