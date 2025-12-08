WITH stats AS (
  SELECT 
    ticker,
    AVG(volatility_7) AS avg_vol,
    STDDEV(volatility_7) AS sd_vol
  FROM `market_data.features`
  GROUP BY ticker
)
SELECT 
  f.date,
  f.ticker,
  f.volatility_7,
  s.avg_vol,
  s.sd_vol,
  (f.volatility_7 - s.avg_vol) / s.sd_vol AS z_score
FROM `market_data.features` f
JOIN stats s USING(ticker)
WHERE (f.volatility_7 - s.avg_vol) / s.sd_vol > 2
ORDER BY z_score DESC;
