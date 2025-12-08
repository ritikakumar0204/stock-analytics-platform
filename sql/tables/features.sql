CREATE TABLE IF NOT EXISTS `market_data.features` (
  date DATE NOT NULL,
  ticker STRING NOT NULL,
  close FLOAT64,
  ma_7 FLOAT64,
  ma_14 FLOAT64,
  ma_21 FLOAT64,
  ema_20 FLOAT64,
  volatility_7 FLOAT64,
  volatility_30 FLOAT64,
  lag_1 FLOAT64,
  lag_7 FLOAT64,
  lag_30 FLOAT64,
  target FLOAT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITION BY date
CLUSTER BY ticker;
