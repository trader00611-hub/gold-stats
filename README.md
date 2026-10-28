# Gold stats

This repository currently contains two small, independent projects:

- [Monthly gold-price drivers model](#gold-price-drivers-a-reproducible-monthly-model): econometric analysis and reproducible FRED/WGC/GPR data workflow.
- [DETR object detection with VITS audio description](object-detection-audio/README.md): Gradio app for annotated detections and spoken scene descriptions.

## Gold price drivers: a reproducible monthly model

An econometric starter project for studying monthly USD gold returns, not predicting a spot-price level or establishing causation. It covers oil, USD/JPY, a broad trade-weighted dollar proxy, US real yields, Fed policy, the Dow, geopolitical risk, and optional ECB/BOJ policy rates and World Gold Council (WGC) supply/demand data.

## Model

Target: monthly log return of the London gold price in USD per troy ounce, log(P[t]) - log(P[t-1]). The baseline is OLS with HAC/Newey-West standard errors. Predictors are lagged one month and expressed as returns or changes: WTI, USD/JPY, broad USD, Dow 30, US 10-year TIPS real yield, Fed funds, optional ECB/BOJ policy rates, and optional geopolitics and WGC variables. Expanding-window forecasts are evaluated out of sample.

Coefficients are conditional associations, not causal effects. A return regression is preferable to a regression of trending price levels on trending macro levels, which can produce spurious fit. The model is for research/education, not investment advice.

## Factors

| Factor | Input / interpretation |
|---|---|
| Oil | WTI monthly log return; energy/inflation and risk channel |
| USD/JPY | Monthly log return; currency and carry/risk sentiment channel |
| DXY | FRED broad trade-weighted USD index DTWEXBGS, a proxy and not ICE DXY |
| Fed cuts / US policy | Change in effective Fed funds; realized easing, not market expectations |
| Real rates | Change in 10-year TIPS real yield, an opportunity-cost measure |
| ECB / BOJ | Optional policy-rate series configured with FRED series IDs |
| US30 | Dow Jones Industrial Average monthly return as a broad equity control |
| Geopolitical tensions | Caldara-Iacoviello monthly GPR index |
| Gold supply and demand | WGC quarterly total supply, total demand, and central-bank net tonnes |
| Central banks | WGC net purchases in tonnes (positive means net buying); demand/supply ratio |

## Data

- FRED daily market and macro series are aggregated to month-end: London gold benchmark GOLDAMGBD228NLBM, WTI DCOILWTICO, USD/JPY DEXJPUS, broad dollar DTWEXBGS, 10-year TIPS real yield DFII10, effective Fed funds FEDFUNDS, and Dow DJIA. FRED API key required; set environment variable FRED_API_KEY.
- ECB / BOJ are optional: set FRED_ECB_SERIES_ID and FRED_BOJ_SERIES_ID to verified series with consistent policy-rate definitions and percentage-point units.
- Download monthly GPR data at https://www.matteoiacoviello.com/gpr.htm and save data/gpr_monthly.csv with columns date,gpr. Cite Caldara and Iacoviello (2022) and the download vintage.
- Use WGC Goldhub data at https://www.gold.org/goldhub/data/gold-demand-by-country and https://www.gold.org/goldhub/data/monthly-central-bank-statistics. Save data/wgc_quarterly.csv with date,central_bank_net_tonnes,total_supply_tonnes,total_demand_tonnes. Values are tonnes and columns may be omitted. Respect WGC download terms and cite the data vintage.
- Quarterly WGC flows are carried forward only after a two-month reporting delay; quarterly buying is not fabricated as monthly purchases. Optional data may reduce sample length.

FRED supplies revised current history, not point-in-time data. Latest GPR readings may be revised. One-month lags reduce look-ahead risk but do not create a true real-time-vintage backtest.

## Run

Requires Python 3.10+, pandas, numpy, requests, statsmodels.

    python -m pip install -r requirements.txt
    $env:FRED_API_KEY = "your_key"
    # Optional: $env:FRED_ECB_SERIES_ID = "verified_series_id"
    # Optional: $env:FRED_BOJ_SERIES_ID = "verified_series_id"
    python model.py

First run downloads FRED observations to data/fred_monthly.csv. Subsequent runs use that file. Add optional CSVs before running to include GPR/WGC. Outputs: results.md and walk_forward_predictions.csv.

## Interpretation and limitations

1. Monthly end-of-period aggregation is used for market series; some FRED rates are monthly averages, so series are not perfectly synchronized.
2. Expected rate cuts differ from realized policy changes. The baseline measures realized changes only; expectations require futures/OIS data.
3. Oil, currencies, dollar, equities, and yields can be correlated. Interpret coefficients with confidence intervals and test stability across subperiods.
4. Geopolitical risk can affect oil, currencies, and real yields; its conditional coefficient is not the total effect.
5. This is not a causal structural model or a trading signal. Out-of-sample results do not guarantee future performance.

## Sources

- FRED API documentation: https://fred.stlouisfed.org/docs/api/fred/overview.html
- World Gold Council Goldhub supply/demand: https://www.gold.org/goldhub/data/gold-demand-by-country
- World Gold Council central-bank data: https://www.gold.org/goldhub/data/monthly-central-bank-statistics
- Caldara & Iacoviello, Measuring Geopolitical Risk, American Economic Review (2022): https://www.matteoiacoviello.com/gpr.htm
