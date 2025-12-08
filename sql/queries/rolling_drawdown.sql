WITH peaks AS (
  SELECT
    date,
    ticker,
    close,
    MAX(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak
  FROM `market_data.prices`
)
SELECT
  date,
  ticker,
  close,
  peak,
  (close - peak) / peak AS drawdown
FROM peaks
ORDER BY drawdown ASC
LIMIT 20;
