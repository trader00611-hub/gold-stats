"""Monthly gold return model with FRED data and optional GPR / WGC inputs."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FRED_SERIES = {
    "gold_usd_oz": "GOLDAMGBD228NLBM", "wti_usd_bbl": "DCOILWTICO",
    "usd_jpy": "DEXJPUS", "broad_usd": "DTWEXBGS",
    "us_10y_real_yield": "DFII10", "fed_funds": "FEDFUNDS", "dow30": "DJIA",
}
OPTIONAL_SERIES = {
    "ecb_policy_rate": os.getenv("FRED_ECB_SERIES_ID", ""),
    "boj_policy_rate": os.getenv("FRED_BOJ_SERIES_ID", ""),
}

def fetch_fred(series_id: str, api_key: str) -> pd.Series:
    response = requests.get("https://api.stlouisfed.org/fred/series/observations", params={
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": "1990-01-01",
    }, timeout=30)
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["observations"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.set_index(pd.to_datetime(frame["date"]))["value"].rename(series_id)

def download_fred() -> pd.DataFrame:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise SystemExit("Set FRED_API_KEY before downloading.")
    series = dict(FRED_SERIES)
    series.update({k: v for k, v in OPTIONAL_SERIES.items() if v})
    daily = pd.concat({name: fetch_fred(code, key) for name, code in series.items()}, axis=1)
    monthly = daily.resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    DATA.mkdir(exist_ok=True)
    monthly.to_csv(DATA / "fred_monthly.csv", index_label="date")
    print(f"Saved {len(monthly)} monthly rows to data/fred_monthly.csv")
    return monthly

def load_inputs() -> pd.DataFrame:
    path = DATA / "fred_monthly.csv"
    if not path.exists():
        return download_fred()
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame = frame.resample("ME").last()
    frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame

def optional_series(frame: pd.DataFrame, filename: str, column: str) -> pd.Series | None:
    path = DATA / filename
    if not path.exists():
        return None
    extra = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if column not in extra:
        raise ValueError(f"{path} must contain a '{column}' column")
    extra = extra[[column]].resample("ME").last()
    extra.index = extra.index.to_period("M").to_timestamp("M")
    return extra[column]

def make_design(raw: pd.DataFrame) -> pd.DataFrame:
    if "gold_usd_oz" not in raw:
        raise ValueError("FRED input needs a gold_usd_oz column")
    x = pd.DataFrame(index=raw.index)
    for col in ["wti_usd_bbl", "usd_jpy", "broad_usd", "dow30"]:
        if col in raw:
            x[f"{col}_return"] = np.log(raw[col].where(raw[col] > 0)).diff()
    for col in ["us_10y_real_yield", "fed_funds", "ecb_policy_rate", "boj_policy_rate"]:
        if col in raw:
            x[f"{col}_change_pp"] = raw[col].diff()
    gpr = optional_series(raw, "gpr_monthly.csv", "gpr")
    if gpr is not None:
        x["gpr_change"] = np.log1p(gpr.clip(lower=0)).diff()
    wgc = optional_series(raw, "wgc_quarterly.csv", "central_bank_net_tonnes")
    if wgc is not None:
        wgc.index = wgc.index + pd.offsets.MonthEnd(2)
        x["central_bank_net_tonnes"] = wgc.reindex(x.index).ffill()
    supply = optional_series(raw, "wgc_quarterly.csv", "total_supply_tonnes")
    demand = optional_series(raw, "wgc_quarterly.csv", "total_demand_tonnes")
    if supply is not None and demand is not None:
        supply.index = supply.index + pd.offsets.MonthEnd(2)
        demand.index = demand.index + pd.offsets.MonthEnd(2)
        x["demand_supply_ratio"] = (demand / supply).reindex(x.index).ffill()
    y = np.log(raw["gold_usd_oz"].where(raw["gold_usd_oz"] > 0)).diff().rename("gold_return")
    model = x.shift(1).join(y).dropna()
    return model.replace([np.inf, -np.inf], np.nan).dropna()

def fit_and_report(model: pd.DataFrame) -> None:
    if len(model) < 120:
        raise ValueError(f"Need at least 120 complete monthly observations; found {len(model)}.")
    y = model["gold_return"]
    X = sm.add_constant(model.drop(columns="gold_return"), has_constant="add")
    split = max(60, int(len(model) * 0.7))
    predictions, actuals = [], []
    for end in range(split, len(model)):
        fit = sm.OLS(y.iloc[:end], X.iloc[:end]).fit()
        predictions.append(float(fit.predict(X.iloc[[end]]).iloc[0]))
        actuals.append(float(y.iloc[end]))
    out = pd.DataFrame({"actual_return": actuals, "predicted_return": predictions}, index=model.index[split:])
    rmse = float(np.sqrt(np.mean((out.actual_return - out.predicted_return) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(out.actual_return ** 2)))
    oos_r2 = 1 - (rmse ** 2 / zero_rmse ** 2) if zero_rmse else float("nan")
    direction = float(np.mean(np.sign(out.actual_return) == np.sign(out.predicted_return)))
    final = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    report = ["# Gold monthly return model results", "",
        f"Sample: {model.index.min():%Y-%m} to {model.index.max():%Y-%m} ({len(model)} months).",
        f"Expanding-window out-of-sample RMSE: {rmse:.5f} monthly log-return units (zero-return RMSE: {zero_rmse:.5f}).",
        f"Out-of-sample R-squared versus zero-return forecast: {oos_r2:.3f}.",
        f"Out-of-sample direction accuracy: {direction:.1%}.",
        "", "Coefficients are conditional associations, not causal effects. Full-sample coefficients are descriptive.",
        "", "## HAC-robust full-sample OLS", "", "    " + final.summary().as_text().replace("\n", "\n    "), "",
        "## Most recent expanding-window predictions", "", out.tail(12).to_string(), ""]
    (ROOT / "results.md").write_text("\n".join(report), encoding="utf-8")
    out.to_csv(ROOT / "walk_forward_predictions.csv", index_label="date")
    print(f"Wrote results.md and walk_forward_predictions.csv; OOS RMSE={rmse:.5f}, direction={direction:.1%}")

def main() -> None:
    fit_and_report(make_design(load_inputs()))

if __name__ == "__main__":
    main()
