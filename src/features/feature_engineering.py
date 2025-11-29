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
logging.info("---- STARTING FEATURE ENGINEERING ETL ----")

# ------------------------------------------
# Load environment variables
# ------------------------------------------
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")

PRICES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.prices"
FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.features"

# ------------------------------------------
# Initialize BigQuery client
# ------------------------------------------
try:
    client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
    logging.info("BigQuery client initialized.")
except Exception as e:
    logging.error(f"BigQuery client init failed: {e}")
    raise


# ------------------------------------------
# Load prices from BigQuery
# ------------------------------------------
def load_prices():
    query = f"""
        SELECT date, open, high, low, close, volume, ticker
        FROM `{PRICES_TABLE}`
        ORDER BY ticker, date
    """
    try:
        df = client.query(query).to_dataframe()
        logging.info(f"Loaded {len(df)} rows from prices table.")
        return df
    except Exception as e:
        logging.error(f"Failed to load price data: {e}")
        raise


# ------------------------------------------
# Compute features for single ticker
# ------------------------------------------
def compute_features(df):
    df = df.copy()
    df = df.sort_values("date")

    # Moving Averages
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_14"] = df["close"].rolling(14).mean()
    df["ma_21"] = df["close"].rolling(21).mean()

    # EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Volatility (rolling standard deviation)
    df["volatility_7"] = df["close"].rolling(7).std()
    df["volatility_30"] = df["close"].rolling(30).std()

    # Lag features
    df["lag_1"] = df["close"].shift(1)
    df["lag_7"] = df["close"].shift(7)
    df["lag_30"] = df["close"].shift(30)

    # Target: next-day close
    df["target"] = df["close"].shift(-1)

    # Remove rows with NaN (from rolling windows)
    df = df.dropna()

    return df


# ------------------------------------------
# Upload to BigQuery
# ------------------------------------------
def upload_features(df):
    try:
        job = client.load_table_from_dataframe(df, FEATURES_TABLE)
        job.result()
        logging.info(f"Uploaded {len(df)} feature rows.")
        print(f"✔ Uploaded {len(df)} rows to {FEATURES_TABLE}")
    except Exception as e:
        logging.error(f"BigQuery upload failed: {e}")
        print("❌ Upload failed. Check etl_logs.log.")
        raise


# ------------------------------------------
# Main ETL
# ------------------------------------------
def main():
    # Step 1: Load raw prices
    prices = load_prices()

    # Step 2: Compute features per ticker
    all_frames = []
    for ticker in prices["ticker"].unique():
        df_t = prices[prices["ticker"] == ticker]
        feats = compute_features(df_t)
        all_frames.append(feats)

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df = final_df[[
    "date", "ticker", "close",
    "ma_7", "ma_14", "ma_21",
    "ema_20",
    "volatility_7", "volatility_30",
    "lag_1", "lag_7", "lag_30",
    "target"
    ]]

    # Step 3: Upload to BigQuery
    upload_features(final_df)

    logging.info("---- FEATURE ENGINEERING ETL COMPLETED ----")
    print("✔ Feature engineering ETL completed.")


if __name__ == "__main__":
    main()
