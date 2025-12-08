CREATE TABLE IF NOT EXISTS `market_data.stocks` (
  ticker STRING NOT NULL,
  company_name STRING,
  sector STRING,
  industry STRING,
  exchange STRING,
  country STRING,
  market_cap INT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
OPTIONS(
  description="Master table containing static metadata about companies"
);
