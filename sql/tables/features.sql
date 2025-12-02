<<<<<<< HEAD
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
=======
CREATE TABLE features (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    ma_7 DECIMAL(12,4),
    ma_30 DECIMAL(12,4),
    ma_90 DECIMAL(12,4),
    rsi DECIMAL(5,2),
    volatility DECIMAL(12,8),
    macd DECIMAL(12,4),
    macd_signal DECIMAL(12,4),
    daily_return DECIMAL(12,8),
    weekly_return DECIMAL(12,8),
    volume_ratio DECIMAL(12,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date), 
    FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE
);
>>>>>>> f88559d (Update features.sql)
