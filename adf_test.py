import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ══════════════════════════════════════════════════════════════════════════
#   STANDALONE ADF STATIONARITY TEST
#   Loads saved CSVs from data_loader cache and runs ADF on log returns
#   for BARC.L, LLOY.L, HSBA.L without rerunning any models.
# ══════════════════════════════════════════════════════════════════════════

FILES = {
    'Barclays': 'saved_data/BARC_L_20210228_20260228.csv',
    'Lloyds':   'saved_data/LLOY_L_20210228_20260228.csv',
    'HSBC':     'saved_data/HSBA_L_20210228_20260228.csv',
}

print("\n" + "═"*60)
print("  ADF STATIONARITY TEST — LOG RETURNS")
print("═"*60)
print(f"  {'Stock':<12} {'ADF Stat':>12} {'p-value':>12} {'Stationary':>12}")
print("  " + "─"*52)

for name, filepath in FILES.items():
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    log_ret = np.log(df[price_col]).diff().dropna()
    adf_stat, p_value, _, _, _, _ = adfuller(log_ret, autolag='AIC')
    stationary = "Yes" if p_value < 0.05 else "No"
    print(f"  {name:<12} {adf_stat:>12.4f} {p_value:>12.6f} {stationary:>12}")

print("═"*60)
print("  Null hypothesis: series has a unit root (non-stationary)")
print("  Reject H0 at p < 0.05 → stationary")
print("═"*60 + "\n")
