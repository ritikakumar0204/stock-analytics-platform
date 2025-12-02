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
