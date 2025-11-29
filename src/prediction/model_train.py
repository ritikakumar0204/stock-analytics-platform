import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
import os
import logging
from sklearn.model_selection import train_test_split
from datetime import timedelta

# ------------------------------------------
# CONFIG
# ------------------------------------------
FORECAST_DAYS = 1    # recursive steps

# MODEL AGNOSTIC CONFIG — CHANGE HERE
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.linear_model import LinearRegression

MODEL_CLASS = RandomForestRegressor
MODEL_PARAMS = {"n_estimators": 200, "random_state": 42}

# ------------------------------------------
# Logging
# ------------------------------------------
logging.basicConfig(
    filename="etl_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("---- STARTING RECURSIVE MODEL-AGNOSTIC PREDICTION ETL ----")

# ------------------------------------------
# Environment
# ------------------------------------------
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("GCP_DATASET_ID")
SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")

FEATURES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.features"
PRED_TABLE = f"{PROJECT_ID}.{DATASET_ID}.predictions"

client = bigquery.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)

# ------------------------------------------
# Load features from BigQuery
# ------------------------------------------
def load_features():
    query = f"""
        SELECT *
        FROM `{FEATURES_TABLE}`
        ORDER BY ticker, date
    """
    df = client.query(query).to_dataframe()
    logging.info(f"Loaded {len(df)} feature rows.")
    return df

# ------------------------------------------
# Update lag features using prediction
# ------------------------------------------
def update_features_with_prediction(row, predicted_close):
    row = row.copy()

    # SHIFT LAGS DOWNWARD
    row["lag_30"] = row["lag_29"] if "lag_29" in row else row["lag_30"]
    row["lag_7"]  = row["lag_6"]  if "lag_6"  in row else row["lag_7"]

    # NEW lag_1 = predicted close
    row["lag_1"] = predicted_close

    # Also update close for approximate recalculation
    row["close"] = predicted_close

    return row

# ------------------------------------------
# Recursive forecasting
# ------------------------------------------
def recursive_forecast(model, last_row, last_date):
    preds = []

    current_features = last_row.copy()
    current_date = pd.to_datetime(last_date)

    for step in range(1, FORECAST_DAYS + 1):
        # Predict one step ahead
        pred = float(model.predict(current_features.to_frame().T)[0])

        preds.append({
            "horizon": step,
            "predicted_close": pred,
            "pred_date": (current_date + timedelta(days=step)).date()
        })

        # Update features for next iteration
        current_features = update_features_with_prediction(current_features, pred)

    return preds

# ------------------------------------------
# Train model for single ticker
# ------------------------------------------
def train_and_forecast(df_ticker):
    df = df_ticker.copy().sort_values("date")

    feature_cols = [
        "close", "ma_7", "ma_14", "ma_21",
        "ema_20", "volatility_7", "volatility_30",
        "lag_1", "lag_7", "lag_30"
    ]

    X = df[feature_cols]
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # MODEL-AGNOSTIC: create model class dynamically
    model = MODEL_CLASS(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    last_row = X.tail(1).iloc[0]
    last_date = df["date"].tail(1).values[0]

    preds = recursive_forecast(model, last_row, last_date)
    return last_date, preds

# ------------------------------------------
# Upload predictions
# ------------------------------------------
def upload_predictions(base_date, ticker, preds):
    df = pd.DataFrame([
        {
            "base_date": base_date,
            "pred_date": p["pred_date"],
            "ticker": ticker,
            "horizon": p["horizon"],
            "predicted_close": p["predicted_close"],
            "model_version": f"{MODEL_CLASS.__name__}_h{FORECAST_DAYS}"
        }
        for p in preds
    ])

    job = client.load_table_from_dataframe(df, PRED_TABLE)
    job.result()

    logging.info(f"Uploaded {len(df)} predictions for {ticker}.")
    print(f"✔ Uploaded {len(df)} predictions for {ticker}")

# ------------------------------------------
# Main ETL
# ------------------------------------------
def main():
    df = load_features()

    for ticker in df["ticker"].unique():
        df_t = df[df["ticker"] == ticker]
        base_date, preds = train_and_forecast(df_t)

        upload_predictions(base_date, ticker, preds)

    logging.info("---- RECURSIVE MODEL-AGNOSTIC PREDICTION ETL COMPLETED ----")
    print("✔ Recursive prediction ETL completed.")


if __name__ == "__main__":
    main()
