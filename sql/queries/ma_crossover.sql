WITH crossovers AS (
  SELECT
    date,
    ticker,
    ma_7,
    ma_21,
    CASE
      WHEN ma_7 > ma_21
           AND LAG(ma_7) OVER (PARTITION BY ticker ORDER BY date)
             < LAG(ma_21) OVER (PARTITION BY ticker ORDER BY date)
      THEN 'Bullish Crossover'
      ELSE NULL
    END AS signal
  FROM `market_data.features`
)

SELECT *
FROM crossovers
WHERE signal IS NOT NULL
ORDER BY date;
