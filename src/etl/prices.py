import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
import os
import logging

# ------------------------------------------
# Logging setup
# ------------------------------------------
logging.basicConfig(
    filename="etl_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("---- STARTING CSV → BIGQUERY UPLOAD ----")

# ------------------------------------------
# Load environment variables
# ------------------------------------------
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")

TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.prices"

# ------------------------------------------
# Initialize BigQuery Client
# ------------------------------------------
try:
    client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
    logging.info("BigQuery client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize BigQuery client: {e}")
    raise

# ------------------------------------------
# Load CSV
# ------------------------------------------
try:
    df = pd.read_csv(r"D:\Code\stock-analytics-platform\dataset\historical_stocks_clean.csv")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

    # Rename to match BigQuery schema
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Ticker": "ticker"
    })

    # Convert types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # df["open"] = pd.to_numeric(df["open"], errors="coerce").astype(float)
    # df["high"] = pd.to_numeric(df["high"], errors="coerce").astype(float)
    # df["low"] = pd.to_numeric(df["low"], errors="coerce").astype(float)
    # df["close"] = pd.to_numeric(df["close"], errors="coerce").astype(float)
    # df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    df["ticker"] = df["ticker"].astype(str)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    logging.info(f"Loaded CSV with {len(df)} rows.")
except Exception as e:
    logging.error(f"Failed to load CSV file: {e}")
    print("❌ Error loading CSV. Check etl_logs.log.")
    raise

# ------------------------------------------
# Upload to BigQuery
# ------------------------------------------
try:
    job = client.load_table_from_dataframe(df, TABLE_ID)
    job.result()
    logging.info(f"Successfully uploaded {len(df)} rows to {TABLE_ID}.")
    print(f"✔ Uploaded {len(df)} rows to {TABLE_ID}")
except Exception as e:
    logging.error(f"BigQuery upload failed: {e}")
    print("❌ BigQuery upload failed. Check etl_logs.log.")
    raise

logging.info("---- CSV → BIGQUERY UPLOAD COMPLETED ----")
