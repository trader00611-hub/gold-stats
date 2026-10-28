# Input file formats

Optional local files are intentionally excluded from version control because they may be downloaded from licensed or revised sources.

## geopolitics: gpr_monthly.csv

Required columns: date,gpr. One observation per month. Source: Caldara-Iacoviello monthly Geopolitical Risk Index. Use official monthly index values and note download vintage.

## World Gold Council: wgc_quarterly.csv

Required date column; supported numeric columns (any subset): central_bank_net_tonnes,total_supply_tonnes,total_demand_tonnes. Dates should identify the reporting quarter. Values are tonnes. WGC quarterly records are lagged two months before being used; the last published observation is carried forward. Cite the WGC source and data vintage and comply with its terms.

## FRED cache: fred_monthly.csv

Generated on the first run using FRED_API_KEY. Do not commit a personal API key. Expected columns include gold_usd_oz,wti_usd_bbl,usd_jpy,broad_usd,us_10y_real_yield,fed_funds,dow30 and optionally ecb_policy_rate,boj_policy_rate.
