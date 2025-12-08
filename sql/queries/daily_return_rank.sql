WITH returns AS (
  SELECT
    ticker,
    date,
    (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
      / LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS daily_return
  FROM `market_data.prices`
)
SELECT
  RANK() OVER (ORDER BY daily_return DESC) AS return_rank,
  ticker,
  date,
  daily_return
FROM returns
WHERE daily_return IS NOT NULL
ORDER BY return_rank
LIMIT 20;
