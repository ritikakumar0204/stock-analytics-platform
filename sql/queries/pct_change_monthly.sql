SELECT 
  ticker,
  FORMAT_DATE('%Y-%m', date) AS month,
  (
    LAST_VALUE(close) OVER wnd - FIRST_VALUE(close) OVER wnd
  ) / FIRST_VALUE(close) OVER wnd AS pct_change_month
FROM `market_data.prices`
WINDOW wnd AS (
  PARTITION BY ticker, FORMAT_DATE('%Y-%m', date)
  ORDER BY date
  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
ORDER BY month, ticker;
