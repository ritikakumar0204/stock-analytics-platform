# !pip install yfinance --quiet
# !pip install google-cloud-bigquery --quiet
# !pip install pandas --quiet
# !pip install pyarrow --quiet

import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from google.cloud import bigquery
import google.auth
import os
from google.colab import files

START_DATE = "2020-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
BQ_PROJECT = "sustained-pod-369913"    # Leave None to auto-detect via auth; or set "your-gcp-project-id"
BQ_DATASET = "stock_forecasting"   # dataset name to create / use in BigQuery
PRICES_TABLE = "prices"
STOCKS_TABLE = "stocks"
FEATURES_TABLE = "features"
MODEL_INPUT_TABLE = "model_input"
PREDICTIONS_TABLE = "predictions"
DOWNLOAD_PAUSE = 0.5

try:
    uploaded = files.upload()  # use the upload UI
    if 'metadata.json' in uploaded:
        metadata = json.loads(uploaded['metadata.json'].decode('utf-8'))
    else:
        first_name = list(uploaded.keys())[0]
        metadata = json.loads(uploaded[first_name].decode('utf-8'))
except Exception as e:
    print("Upload skipped or failed:", e)

stocks_df = pd.DataFrame(metadata)
if 'ticker' not in stocks_df.columns:
    raise ValueError("metadata.json must contain 'ticker' for each entry.")
tickers = stocks_df['ticker'].unique().tolist()
print(f"Tickers to download ({len(tickers)}):", tickers)

def download_ticker_history(ticker, start, end, max_retries=3):
    attempt = 0
    last_err = None
    while attempt < max_retries:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return pd.DataFrame()
            df.columns = df.columns.droplevel(1)
            df = df.reset_index().rename(columns={'Date':'date', 'Adj Close':'adj_close'})
            df['ticker'] = ticker
            df = df.rename(columns={c: c.strip().lower().replace(' ', '_') for c in df.columns})
            keep_cols = [c for c in ['date','open','high','low','close','adj_close','volume','ticker'] if c in df.columns]
            return df[keep_cols]
        except Exception as e:
            last_err = e
            attempt += 1
            time.sleep(1 + attempt)
    print(f"Failed to download {ticker} after {max_retries} attempts. Last error:", last_err)
    return pd.DataFrame()

prices_frames = []
start_time = time.time()
for i, t in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] downloading {t} ...", end=' ')
    df_t = download_ticker_history(t, START_DATE, END_DATE)
    if df_t.empty:
        print("NO DATA")
    else:
        print(f"rows={len(df_t)}")
        prices_frames.append(df_t)
    time.sleep(DOWNLOAD_PAUSE)

prices_df = pd.concat(prices_frames, ignore_index=True).sort_values(['ticker','date']).reset_index(drop=True)
prices_df['date'] = pd.to_datetime(prices_df['date'])
for col in ['open','high','low','close','adj_close','volume']:
    if col in prices_df.columns:
        prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce')
prices_df.tail()

prices_df = prices_df.dropna(subset=['date','adj_close']).reset_index(drop=True)

summary = prices_df.groupby('ticker').agg(start_date=('date','min'), end_date=('date','max'), rows=('date','count')).reset_index()
print("\nPer-ticker download summary:")
display(summary)

from google.colab import auth
auth.authenticate_user()
creds, project = google.auth.default()
if BQ_PROJECT is None:
    BQ_PROJECT = project
print("Using GCP project:", BQ_PROJECT)

client = bigquery.Client(project=BQ_PROJECT)
dataset_ref = client.dataset(BQ_DATASET)
try:
    client.get_dataset(dataset_ref)  # raises NotFound if dataset does not exist
    print(f"Dataset {BQ_PROJECT}.{BQ_DATASET} already exists.")
except Exception:
    print(f"Creating dataset {BQ_PROJECT}.{BQ_DATASET}")
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset)  # API request

stocks_out = stocks_df.copy()
stocks_out.columns = [c.strip().lower().replace(' ', '_') for c in stocks_out.columns]

table_ref = dataset_ref.table(STOCKS_TABLE)
job = client.load_table_from_dataframe(stocks_out, table_ref, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
job.result()

prices_out = prices_df.copy()
prices_out.columns = [c.strip().lower().replace(' ', '_') for c in prices_out.columns]
WRITE_MODE = "WRITE_TRUNCATE"  # change to WRITE_APPEND for incremental
print("\nUploading prices to BigQuery table:", PRICES_TABLE, "mode:", WRITE_MODE)

for t in prices_out['ticker'].unique():
    df_t = prices_out[prices_out['ticker'] == t].reset_index(drop=True)
    table_ref = dataset_ref.table(PRICES_TABLE)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND" if WRITE_MODE=="WRITE_APPEND" else "WRITE_TRUNCATE")
    # For the first ticker with WRITE_TRUNCATE, use TRUNCATE; then switch to APPEND
    if WRITE_MODE == "WRITE_TRUNCATE":
        # first run will truncate, subsequent must append
        job = client.load_table_from_dataframe(df_t, table_ref, job_config=job_config)
        job.result()
        WRITE_MODE = "WRITE_APPEND"
    else:
        job = client.load_table_from_dataframe(df_t, table_ref, job_config=job_config)
        job.result()
    print(f"Uploaded ticker {t} rows={len(df_t)}")
print("Prices upload complete.")

print(f"Project: {BQ_PROJECT}, Dataset: {BQ_DATASET}, Tables: {STOCKS_TABLE}, {PRICES_TABLE}")

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['adj_close'].shift(1)).abs()
    low_close = (df['low'] - df['adj_close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    return atr

tickers_df = client.query(f"SELECT DISTINCT ticker, market_cap FROM `{BQ_PROJECT}.{BQ_DATASET}.{STOCKS_TABLE}` LIMIT 200").to_dataframe()
tickers = tickers_df['ticker'].tolist()
marketcap_map = tickers_df.set_index('ticker')['market_cap'].to_dict()
print("Tickers:", len(tickers))

tickers.append('AAPL')
tickers.append('MSFT')
# tickers.append('AMZN')
tickers

q = f"""
SELECT ticker, DATE(date) AS date, open, high, low, close, adj_close, volume
FROM `{BQ_PROJECT}.{BQ_DATASET}.{PRICES_TABLE}`
WHERE ticker IS NOT NULL
ORDER BY ticker, date
"""
prices = client.query(q).to_dataframe()
if prices.empty:
    raise RuntimeError("No price rows returned. Check PRICES_TABLE name and dataset.")
prices['date'] = pd.to_datetime(prices['date'])
print("Price rows:", len(prices))

frames = []
for t in tickers:
    df = prices[prices['ticker']==t].sort_values('date').reset_index(drop=True)
    if df.empty:
        continue
    # core transforms
    df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df['return_1d'] = df['adj_close'].pct_change()
    # moving avgs
    df['ma_5'] = df['adj_close'].rolling(5, min_periods=1).mean()
    df['ma_10'] = df['adj_close'].rolling(10, min_periods=1).mean()
    df['ma_20'] = df['adj_close'].rolling(20, min_periods=1).mean()
    df['ma_50'] = df['adj_close'].rolling(50, min_periods=1).mean()
    # EMA and MACD
    df['ema_12'] = ema(df['adj_close'], span=12)
    df['ema_26'] = ema(df['adj_close'], span=26)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = ema(df['macd'], span=9)
    df['ma_10_minus_ma_50'] = df['ma_10'] - df['ma_50']
    # RSI, ATR
    df['rsi_14'] = compute_rsi(df['adj_close'], window=14)
    df['atr_14'] = compute_atr(df, n=14)
    # Volatility (annualized)
    df['vol_10'] = df['log_return'].rolling(10, min_periods=1).std() * np.sqrt(252)
    df['vol_20'] = df['log_return'].rolling(20, min_periods=1).std() * np.sqrt(252)
    df['vol_60'] = df['log_return'].rolling(60, min_periods=1).std() * np.sqrt(252)
    # Momentum
    df['mom_1m'] = df['adj_close'].pct_change(21)
    df['mom_3m'] = df['adj_close'].pct_change(63)
    df['mom_6m'] = df['adj_close'].pct_change(126)
    df['mom_12m'] = df['adj_close'].shift(21).pct_change(252)  # approximate skip last month
    # Volume
    df['avg_volume_20'] = df['volume'].rolling(20, min_periods=1).mean()
    # Calendar
    df['day_of_week'] = df['date'].dt.weekday
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    # Market cap from stocks table if present
    df['market_cap'] = df['ticker'].map(marketcap_map)
    df['log_market_cap'] = np.log(df['market_cap'].replace(0, np.nan))
    frames.append(df)

features_df = pd.concat(frames, ignore_index=True, sort=False)
print("Computed technical features rows:", len(features_df))

cs_cols = ['mom_3m','vol_20','avg_volume_20']
for col in cs_cols:
    if col not in features_df.columns:
        features_df[col] = np.nan
    features_df[f"{col}_z"] = features_df.groupby('date')[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    features_df[f"{col}_rankpct"] = features_df.groupby('date')[col].rank(pct=True, method='average')

features_df = features_df.sort_values(['ticker','date']).reset_index(drop=True)
features_df['target_return_1d'] = features_df.groupby('ticker')['adj_close'].shift(-1) / features_df['adj_close'] - 1
features_df['direction_1d'] = (features_df['target_return_1d'] > 0).astype(int)
features_df['target_return_5d'] = features_df.groupby('ticker')['adj_close'].shift(-5) / features_df['adj_close'] - 1

feature_version = "tech_only_v1_" + datetime.utcnow().strftime("%Y%m%dT%H%MZ")
features_df['feature_version'] = feature_version
features_df['source'] = "yfinance_price"
features_df['computed_at'] = datetime.utcnow()

final_cols = [
 'ticker','date','adj_close','open','high','low','close','volume',
 'log_return','return_1d',
 'ma_5','ma_10','ma_20','ma_50','ema_12','ema_26','ma_10_minus_ma_50','macd','macd_signal',
 'rsi_14','atr_14',
 'vol_10','vol_20','vol_60',
 'mom_1m','mom_3m','mom_6m','mom_12m',
 'avg_volume_20','log_market_cap','day_of_week','is_month_end',
 'mom_3m_z','mom_3m_rankpct','vol_20_z','vol_20_rankpct',
 'target_return_1d','direction_1d','target_return_5d',
 'feature_version','source','computed_at'
]
final_cols = [c for c in final_cols if c in features_df.columns]
model_input = features_df[final_cols].copy()
# drop last rows per ticker where target is NaN (no next day)
model_input = model_input.dropna(subset=['target_return_1d']).reset_index(drop=True)
print("Final model_input rows:", len(model_input))

from google.cloud.bigquery import LoadJobConfig

load_config = LoadJobConfig(write_disposition="WRITE_TRUNCATE")
client.load_table_from_dataframe(features_df, f"{BQ_PROJECT}.{BQ_DATASET}.{FEATURES_TABLE}", job_config=load_config).result()
print("Uploaded features rows:", client.get_table(f"{BQ_PROJECT}.{BQ_DATASET}.{FEATURES_TABLE}").num_rows)

print("Uploading model_input table to BigQuery:", MODEL_INPUT_TABLE)
client.load_table_from_dataframe(model_input, f"{BQ_PROJECT}.{BQ_DATASET}.{MODEL_INPUT_TABLE}", job_config=load_config).result()
print("Uploaded model_input rows:", client.get_table(f"{BQ_PROJECT}.{BQ_DATASET}.{MODEL_INPUT_TABLE}").num_rows)

# ---------- create empty predictions table schema ----------
print("Creating/Resetting empty predictions table:", PREDICTIONS_TABLE)
from google.cloud.bigquery import SchemaField, Table
schema = [
    SchemaField("ticker","STRING"),
    SchemaField("date","DATE"),
    SchemaField("model_name","STRING"),
    SchemaField("predicted", "FLOAT64"),
    SchemaField("predicted_return", "FLOAT64"),
    SchemaField("confidence", "FLOAT64"),
    SchemaField("accuracy", "FLOAT64"),
    SchemaField("created_at","TIMESTAMP"),
]
table_ref = client.dataset(BQ_DATASET).table(PREDICTIONS_TABLE)
# Delete if exists
try:
    client.delete_table(table_ref)
except Exception:
    pass
table = Table(table_ref, schema=schema)
table = client.create_table(table)
print("Created predictions table:", table.table_id)

print("All done — model_input and features uploaded. Feature version:", feature_version)
