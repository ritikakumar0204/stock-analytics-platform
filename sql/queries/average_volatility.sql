SELECT 
  ticker,
  AVG(volatility_7) AS avg_volatility_7,
  AVG(volatility_30) AS avg_volatility_30
FROM `market_data.features`
GROUP BY ticker
ORDER BY avg_volatility_30 DESC;
