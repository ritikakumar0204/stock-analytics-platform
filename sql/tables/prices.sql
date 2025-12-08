CREATE TABLE IF NOT EXISTS `market_data.prices` (
  ticker STRING NOT NULL,
  date DATE NOT NULL,
  timestamp TIMESTAMP,
  open FLOAT64,
  high FLOAT64,
  low FLOAT64,
  close FLOAT64,
  volume INT64,
  pct_change FLOAT64,
  daily_return FLOAT64,
  source STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITION BY date
CLUSTER BY ticker
OPTIONS(
  description="Historical OHLCV price data with engineered features."
);
