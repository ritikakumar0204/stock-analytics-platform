WITH base AS (
  SELECT
    date,
    ticker,
    close
  FROM `market_data.prices`
),

pairs AS (
  SELECT
    a.ticker AS ticker_a,
    b.ticker AS ticker_b
  FROM (SELECT DISTINCT ticker FROM base) a
  JOIN (SELECT DISTINCT ticker FROM base) b
    ON a.ticker < b.ticker    -- avoid duplicates (AAPL–MSFT only once)
)

SELECT
  p.ticker_a,
  p.ticker_b,
  CORR(a.close, b.close) AS correlation
FROM pairs p
JOIN base a ON a.ticker = p.ticker_a
JOIN base b ON b.ticker = p.ticker_b
           AND a.date = b.date         -- align dates
GROUP BY
  p.ticker_a,
  p.ticker_b
ORDER BY
  ABS(correlation) DESC;
