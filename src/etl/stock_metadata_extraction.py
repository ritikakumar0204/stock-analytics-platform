import json
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
import os
import logging
from datetime import datetime


# Load environment variables

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")

TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.stocks"


# Logging setup

logging.basicConfig(
    filename="etl_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("STARTING METADATA TO BIGQUERY ETL")


# Initialize BigQuery Client

try:
    client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
    logging.info("BigQuery client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize BigQuery client: {e}")
    raise e



# Load metadata JSON

def load_metadata_json(path=r"D:\Code\stock-analytics-platform\dataset\metadata.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)

        if not data:
            raise ValueError("metadata.json is empty")

        logging.info(f"Successfully loaded metadata.json with {len(data)} records")
        return data

    except FileNotFoundError:
        logging.error("metadata.json file not found.")
        raise

    except Exception as e:
        logging.error(f"Error reading metadata.json: {e}")
        raise


# Upload DataFrame to BigQuery

def upload_to_bigquery(df):
    try:
        job = client.load_table_from_dataframe(df, TABLE_ID)
        job.result()  # Wait for completion

        logging.info(
            f"Uploaded {len(df)} rows to BigQuery table {TABLE_ID} successfully."
        )
        print(f"Uploaded {len(df)} rows to BigQuery table {TABLE_ID}. ✔")

    except Exception as e:
        logging.error(f"BigQuery upload failed: {e}")
        print("❌ BigQuery upload failed. Check etl_logs.log.")
        raise



# Main ETL Flow

def main():
    start_time = datetime.now()

    # Step 1 — load JSON file
    try:
        metadata = load_metadata_json()
    except Exception as e:
        print("❌ Error loading metadata.json. ETL aborted.")
        return

    # Step 2 — convert to DataFrame
    try:
        df = pd.DataFrame(metadata)

        # Validate columns
        REQUIRED_COLS = [
            "ticker",
            "company_name",
            "sector",
            "industry",
            "exchange",
            "country",
            "market_cap",
        ]

        if not all(col in df.columns for col in REQUIRED_COLS):
            missing = [col for col in REQUIRED_COLS if col not in df.columns]
            logging.error(f"Missing required columns: {missing}")
            print(f"Missing columns in metadata.csv: {missing}")
            return

        logging.info("Metadata successfully converted to DataFrame.")

    except Exception as e:
        logging.error(f"Failed to convert metadata to DataFrame: {e}")
        print("Error converting metadata to DataFrame.")
        return

    # Step 3 — upload to BigQuery
    try:
        upload_to_bigquery(df)
    except Exception:
        return

    # Finished ETL
    end_time = datetime.now()
    duration = (end_time - start_time).seconds

    logging.info(
        f"METADATA ETL COMPLETED in {duration} seconds"
    )
    print("Metadata ETL completed successfully.")


# ------------------------------------------
# Execute Script
# ------------------------------------------
if __name__ == "__main__":
    main()
