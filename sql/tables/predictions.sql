CREATE TABLE IF NOT EXISTS `market_data.predictions` (
  base_date DATE NOT NULL,           -- last date in features table
  pred_date DATE NOT NULL,           -- the future date being predicted
  ticker STRING NOT NULL,            -- AAPL, MSFT, etc.
  horizon INT64 NOT NULL,            -- 1 to 7
  predicted_close FLOAT64,           -- model output
  model_version STRING,              -- e.g. RandomForestRegressor_h7
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITION BY base_date
CLUSTER BY ticker;
