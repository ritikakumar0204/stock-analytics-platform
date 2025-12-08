from google.cloud import bigquery
import yfinance as yf
import pandas as pd
import datetime as dt
import logging
from google.cloud import bigquery
from dotenv import load_dotenv
import os
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.prices"

client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)

def run_query(sql: str, project_id: str):
    try:
        client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
        logging.info("BigQuery client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize BigQuery client: {e}")
        raise

    query_job = client.query(sql)
    result = query_job.result()  # Waits for job completion.
    return result.to_dataframe()

if __name__ == "__main__":
    project_id = PROJECT_ID

    sql = """
        SELECT
            *
        from market_data.prices
        limit 1;
    """

    df = run_query(sql, project_id)
    print(df)
