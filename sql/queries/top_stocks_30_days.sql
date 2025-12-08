WITH recent AS (
  SELECT 
    ticker,
    date,
    close,
    FIRST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS close_30_days_ago
  FROM `market_data.prices`
)
SELECT 
  ticker,
  (close - close_30_days_ago) / close_30_days_ago AS pct_return_30d
FROM recent
WHERE close_30_days_ago IS NOT NULL
ORDER BY pct_return_30d DESC;
