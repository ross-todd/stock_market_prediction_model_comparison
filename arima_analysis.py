# ═════════════════════════════════════════════════════════════════════════════════
#   Ross Todd
#   BSc (Hons) Software Development
#   Honours Project 2026 - Stock Market Prediction Comparison Analysis
#
#   ARIMA model — walk-forward validation and 5-day forecast for BARC, LLOY, HSBA
#
#   In this project, ARIMA is used as a univariate time-series baseline,
#   trained on log-returns only. Random Forest and GRU are both trained as
#   multivariate models with engineered technical features. ARIMA provides a 
#   classical statistical baseline against which the advanced feature-based models 
#   can be benchmarked. All three share the same data window, 80/20 split,
#   walk-forward protocol, evaluation metrics, and refit frequency 
#   (every 63 trading days) so that differences in predictive performance are 
#   attributable to model class rather than evaluation methodology.
# ═════════════════════════════════════════════════════════════════════════════════

import pandas as pd
from data_loader import load_all_tickers
import os
import sys
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from scipy.stats import norm
from tabulate import tabulate

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════
#   OUTPUT FOLDERS
#   These are the folders where everything gets saved to. There is
#   one folder each for plots, per-ticker CSVs, and summary tables, all
#   sitting inside the main results folder. They get created automatically
#   if they don't exist yet so the script doesn't fail on a fresh run.
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_FOLDER     = "arima_results"
PER_TICKER_FOLDER = os.path.join(OUTPUT_FOLDER, "per_ticker_results")
SUMMARY_FOLDER    = os.path.join(OUTPUT_FOLDER, "summary")

for folder in [OUTPUT_FOLDER, PER_TICKER_FOLDER, SUMMARY_FOLDER]:
    os.makedirs(folder, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#   TERMINAL OUTPUT LOGGING
#   This saves everything that gets printed to the console into a text
#   file at the same time, so I have a permanent record of every run.
#   The Tee class writes to both the screen and the log file at once.
#   At the very end of the script, stdout gets switched back to normal.
# ══════════════════════════════════════════════════════════════════════════

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file     = open(os.path.join(OUTPUT_FOLDER, "terminal_output.txt"), "w", encoding="utf-8")
_real_stdout = sys.stdout
sys.stdout   = Tee(_real_stdout, log_file)


# ══════════════════════════════════════════════════════════════════════════
#   DATE RANGE & SPLIT CONFIGURATION
#   This sets the five year data window that all three models (ARIMA, RF,
#   GRU) use, along with the 80/20 train/test split. Keeping everything
#   the same across models means any difference in results is down to the
#   model itself, not the data it was trained or tested on.
# ══════════════════════════════════════════════════════════════════════════

TRAIN_RATIO = 0.80
DATA_START, DATA_END = "2021-02-28", "2026-02-28"


# ══════════════════════════════════════════════════════════════════════════
#   TICKERS & MODEL SETTINGS
#   The three UK banking stocks I'm analysing, plus all the ARIMA settings.
#   P_MAX and Q_MAX set how far the grid search goes when trying different
#   model orders. VAL_SIZE is one quarter of trading days (63 trading days), 
#   used to pick the best configuration before running the full test. 
#   WALK_REFIT_FREQ controls how often the model gets retrained during
#   walk-forward. WINDOW_SIZES lets the search also try limiting 
#   how much history the model sees at each refit, rather than always using 
#   all of it.
# ══════════════════════════════════════════════════════════════════════════

TICKERS      = ['BARC.L', 'LLOY.L', 'HSBA.L']
P_MAX, Q_MAX = 3, 3
TRENDS       = ['n', 'c']
VAL_SIZE     = 63
WINDOW_SIZES = [21, 63, 126, 252, None]
WALK_REFIT_FREQ = 63


# ══════════════════════════════════════════════════════════════════════════
#   HELPER FUNCTIONS
#   A few utility functions used throughout the script.
#   check_stationarity runs an ADF test and tries differencing up to twice
#   if the series isn't stationary yet, returning the d value needed.
#   diebold_mariano_test checks whether ARIMA is actually doing better than
#   just guessing tomorrow will be the same as today (the naive baseline).
#   winkler_score measures how good the prediction intervals are — it
#   penalises both wide intervals and ones that miss the actual value.
#   winkler_score_normalised is the same thing but divided by the average
#   price, so I can fairly compare the three stocks against each other
#   even though they trade at very different price levels.
# ══════════════════════════════════════════════════════════════════════════

def check_stationarity(series, max_diff=2):
    series_copy = series.dropna().copy()
    for d in range(max_diff + 1):
        if d > 0:
            series_copy = series_copy.diff().dropna()
        if len(series_copy) < 10:
            break
        adf_result = adfuller(series_copy, autolag='AIC')
        if adf_result[1] <= 0.05:
            return d, adf_result[1]
    return max_diff, 1.0


def diebold_mariano_test(errors1, errors2):
    errors1 = np.asarray(errors1)
    errors2 = np.asarray(errors2)
    loss_differential = errors1**2 - errors2**2
    dm_stat  = np.mean(loss_differential) / np.sqrt(np.var(loss_differential, ddof=1) / len(loss_differential))
    p_value  = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def winkler_score(actual, lower, upper, alpha=0.05):
    # Raw Winkler score in price units (GBX - Great British Pence). Use winkler_score_normalised
    # for cross-ticker comparison — divides by mean actual price.
    actual  = np.asarray(actual)
    lower   = np.asarray(lower)
    upper   = np.asarray(upper)
    width   = upper - lower
    penalty = np.where(actual < lower,
                       2 / alpha * (lower - actual),
                       np.where(actual > upper,
                                2 / alpha * (actual - upper),
                                0.0))
    return np.mean(width + penalty)


def winkler_score_normalised(actual, lower, upper, alpha=0.05):
    # ══════════════════════════════════════════════════════════════════════
    #   NORMALISED WINKLER SCORE
    #   Divides the raw Winkler score by the mean actual price to produce
    #   a scale-free metric expressed as a proportion of the average price.
    #   This allows valid cross-ticker comparison between BARC (~452p),
    #   LLOY (~102p), and HSBA (~1393p), which the raw score cannot support
    #   because interval widths naturally scale with price level.
    #   A lower normalised score still indicates better interval quality.
    #   Formula: Winkler_norm = Winkler_raw / mean(actual)
    # ══════════════════════════════════════════════════════════════════════
    raw_winkler = winkler_score(actual, lower, upper, alpha)
    mean_price  = np.mean(np.asarray(actual))
    return raw_winkler / mean_price if mean_price != 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════
#   MAIN ANALYSIS LOOP
#   This is where everything actually runs. For each ticker, the script
#   loads the data, checks stationarity, does the train/test split, runs
#   a grid search to find the best ARIMA order, then does walk-forward
#   validation to get realistic out-of-sample predictions. After that it
#   calculates all the performance metrics, generates the plots, and
#   produces the 5 day forecast. Results are saved as CSV files.
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*100)
print("ARIMA ANALYSIS - UK BANKING STOCKS (NEXT-DAY WALK-FORWARD + 5-DAY FORECAST)".center(100))
print("═"*100)
print(f"  Data window : {DATA_START}  →  {DATA_END}  (5 calendar years, 80/20 split)")
print(f"  Split       : 80% train / 20% test  |  Val window: {VAL_SIZE} obs")
print(f"  Walk-forward refit frequency: every {WALK_REFIT_FREQ} trading days (1 quarter)")
print("═"*100 + "\n")

print("[DATA] Loading all tickers via data_loader...\n")
ticker_data = load_all_tickers(TICKERS, DATA_START, DATA_END, verbose=True)
print()

all_performance_rows    = []
all_forecasts           = []
all_uncertainty_metrics = []

for ticker in TICKERS:
    if ticker not in ticker_data:
        print(f" Skipping {ticker} – download failed.")
        continue

    print("\n" + "═"*100)
    print(f"   PROCESSING: {ticker}   ".center(100, "═"))
    print("═"*100 + "\n")

    print(f"[DATA] Data loaded for {ticker}...")
    raw_data = ticker_data[ticker]

    available_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                      if c in raw_data.columns]
    inspect_df = raw_data[available_cols].copy()
    inspect_df.index = inspect_df.index.date
    snapshot = pd.concat([inspect_df.head(5), inspect_df.tail(5)])
    print(f"\n   Data Preview (first 5 / last 5 rows):")
    print(tabulate(snapshot.reset_index(), headers=['Date'] + list(snapshot.columns),
                   tablefmt='simple', floatfmt='.3f'))
    print(f"   Total observations: {len(raw_data)}")
    print(f"   Date range: {raw_data.index[0].date()} → {raw_data.index[-1].date()}")

    print(f"\n[SPLIT] Preprocessing data and applying train/test split...")
    df = raw_data.copy().dropna()
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

    df['Log_Ret'] = np.log(df[price_col]).diff()
    df = df.dropna()

    integration_order, adf_pval = check_stationarity(df['Log_Ret'])
    print(f"   Stationarity test: d={integration_order},  ADF p-value={adf_pval:.6f}")
    print(f"   (Log returns are typically stationary at d=0)")

    if len(df) < 10:
        print(f"   ✗ Insufficient data for {ticker}"); continue

    split_idx = int(len(df) * TRAIN_RATIO)

    train_log_returns = df['Log_Ret'].iloc[:split_idx]
    test_log_returns  = df['Log_Ret'].iloc[split_idx:]
    train_prices      = df[price_col].iloc[:split_idx]
    test_prices       = df[price_col].iloc[split_idx:]

    print(f"   Train : {len(train_log_returns)} obs  "
          f"({train_log_returns.index[0].date()} → {train_log_returns.index[-1].date()})")
    print(f"   Test  : {len(test_log_returns)} obs  "
          f"({test_log_returns.index[0].date()} → {test_log_returns.index[-1].date()})")
    print(f"   Ratio : {len(train_log_returns)/(len(train_log_returns)+len(test_log_returns)):.4f} / "
          f"{len(test_log_returns)/(len(train_log_returns)+len(test_log_returns)):.4f}")

    fallback_std = np.std(train_log_returns.values)

    print(f"\n[SEARCH] Running grid search for optimal ARIMA configuration...")

    val_log_returns = train_log_returns[-VAL_SIZE:]
    val_prices      = train_prices[-VAL_SIZE:]
    val_base_price  = train_prices.iloc[-(VAL_SIZE + 1)]

    total_combos = (P_MAX + 1) * (Q_MAX + 1) * len(WINDOW_SIZES) * len(TRENDS)
    print(f"   Testing {total_combos} combinations sequentially...")
    print(f"   Validation window: {VAL_SIZE} obs ({val_log_returns.index[0].date()} → {val_log_returns.index[-1].date()})")

    pretrain_log_returns = list(train_log_returns[:-VAL_SIZE].values)
    val_log_ret_vals     = val_log_returns.values
    val_prices_vals      = val_prices.values

    def _evaluate_combo(p, q, trend, history_window, d):
        warnings.filterwarnings("ignore")
        try:
            rolling_history  = list(pretrain_log_returns)
            current_price    = val_base_price
            predicted_prices = []
            for i in range(VAL_SIZE):
                history_subset       = rolling_history if history_window is None else rolling_history[-history_window:]
                candidate_model      = ARIMA(history_subset, order=(p, d, q), trend=trend).fit()
                predicted_log_return = candidate_model.forecast(steps=1)[0]
                next_price           = current_price * np.exp(predicted_log_return)
                predicted_prices.append(next_price)
                rolling_history.append(val_log_ret_vals[i])
                current_price = val_prices_vals[i]
            val_mape = mean_absolute_percentage_error(val_prices_vals, np.array(predicted_prices))
            return {'p': p, 'd': d, 'q': q, 'trend': trend,
                    'win': history_window, 'val_mape': val_mape}
        except:
            return None

    combos = [
        (p, q, trend, history_window)
        for history_window in WINDOW_SIZES
        for trend          in TRENDS
        for p              in range(P_MAX + 1)
        for q              in range(Q_MAX + 1)
    ]

    raw_results = [
        _evaluate_combo(p, q, trend, history_window, integration_order)
        for p, q, trend, history_window in combos
    ]

    all_combinations = []
    best_val_mape    = float('inf')
    best_config      = {}

    for combo_result in raw_results:
        if combo_result is None:
            continue
        all_combinations.append({
            'Ticker': ticker,
            'p': combo_result['p'], 'd': combo_result['d'], 'q': combo_result['q'],
            'Trend':    combo_result['trend'],
            'Window':   combo_result['win'] if combo_result['win'] else 'Full',
            'Val_MAPE': combo_result['val_mape'],
            'Order':    f"({combo_result['p']},{combo_result['d']},{combo_result['q']})"})
        if combo_result['val_mape'] < best_val_mape:
            best_val_mape = combo_result['val_mape']
            best_config   = {
                'order':    (combo_result['p'], combo_result['d'], combo_result['q']),
                'trend':    combo_result['trend'],
                'window':   combo_result['win'],
                'val_mape': combo_result['val_mape']}

    print(f"   - Grid search complete: {len(all_combinations)} valid configurations")

    combo_df = pd.DataFrame(all_combinations).sort_values('Val_MAPE')
    combo_df.to_csv(os.path.join(PER_TICKER_FOLDER,
        f"{ticker}_grid_search_arima.csv"), index=False)
    print(f"   - Saved: {ticker}_grid_search_arima.csv")
    print(f"\n   Top 5 Configurations by Validation MAPE:")
    print(tabulate(combo_df.head(5)[['Order', 'Trend', 'Window', 'Val_MAPE']],
                   headers='keys', tablefmt='simple', showindex=False, floatfmt='.6f'))

    print(f"\n[MODEL] Training final model with best configuration...")
    best_order  = best_config['order']
    best_trend  = best_config['trend']
    best_window = best_config['window']
    print(f"   Best model: ARIMA{best_order},  Trend={best_trend},  Window={best_window if best_window else 'Full'}")
    print(f"   Validation MAPE: {best_config['val_mape']:.6f}")

    if best_window is None:
        final_train_returns = train_log_returns
    else:
        final_train_returns = train_log_returns.iloc[-best_window:]

    final_model = ARIMA(final_train_returns, order=best_order, trend=best_trend).fit()

    pd.DataFrame([{'Ticker': ticker, 'Model': 'ARIMA', 'Order': str(best_order),
                   'Trend': best_trend, 'Window': best_window if best_window else 'Full',
                   'Val_MAPE': best_config['val_mape'],
                   'AIC': final_model.aic, 'BIC': final_model.bic}]) \
      .to_csv(os.path.join(PER_TICKER_FOLDER,
          f"{ticker}_best_hyperparameters.csv"), index=False)
    print(f"   - Saved: {ticker}_best_hyperparameters.csv")

    print(f"\n[WALK-FORWARD] Running walk-forward validation "
          f"(refit every {WALK_REFIT_FREQ} days)...")

    rolling_history = list(train_log_returns.values)
    prev_prices_wf  = np.concatenate([[train_prices.iloc[-1]], test_prices.values[:-1]])

    wf_predicted_logrets = []
    wf_ci_lower_logrets  = []
    wf_ci_upper_logrets  = []
    skipped_steps        = 0
    current_model        = final_model

    for step_idx in range(len(test_log_returns)):

        if step_idx > 0 and step_idx % WALK_REFIT_FREQ == 0:
            refit_subset = np.array(rolling_history if best_window is None else rolling_history[-best_window:])
            if (len(refit_subset) >= 10 and np.std(refit_subset) >= 1e-8
                    and np.all(np.isfinite(refit_subset))):
                try:
                    current_model = ARIMA(refit_subset, order=best_order, trend=best_trend).fit()
                except Exception:
                    pass

        history_subset = np.array(rolling_history if best_window is None else rolling_history[-best_window:])

        if len(history_subset) < 10 or np.std(history_subset) < 1e-8 or not np.all(np.isfinite(history_subset)):
            wf_predicted_logrets.append(0.0)
            wf_ci_lower_logrets.append(-3.0 * fallback_std)
            wf_ci_upper_logrets.append( 3.0 * fallback_std)
            rolling_history.append(test_log_returns.iloc[step_idx])
            skipped_steps += 1
            continue

        try:
            forecast_obj     = current_model.get_forecast(steps=1, alpha=0.05)
            predicted_logret = float(np.asarray(forecast_obj.predicted_mean)[0])
            ci_bounds        = forecast_obj.conf_int()

            if isinstance(ci_bounds, pd.DataFrame):
                ci_lower_val, ci_upper_val = ci_bounds.iloc[0, 0], ci_bounds.iloc[0, 1]
            else:
                ci_lower_val, ci_upper_val = float(ci_bounds[0, 0]), float(ci_bounds[0, 1])

            if not np.isfinite(predicted_logret) or not np.isfinite(ci_lower_val) or not np.isfinite(ci_upper_val):
                raise ValueError("Non-finite forecast output")

            wf_predicted_logrets.append(predicted_logret)
            wf_ci_lower_logrets.append(float(ci_lower_val))
            wf_ci_upper_logrets.append(float(ci_upper_val))

        except Exception:
            wf_predicted_logrets.append(0.0)
            wf_ci_lower_logrets.append(-3.0 * fallback_std)
            wf_ci_upper_logrets.append( 3.0 * fallback_std)
            skipped_steps += 1

        rolling_history.append(test_log_returns.iloc[step_idx])

        if (step_idx + 1) % 50 == 0:
            print(f"   Progress: {step_idx+1}/{len(test_log_returns)} predictions... (skipped: {skipped_steps})")

    if skipped_steps > 0:
        print(f"    {skipped_steps} windows skipped due to numerical instability — wide fallback CI used")

    wf_predicted_prices = prev_prices_wf * np.exp(wf_predicted_logrets)
    wf_ci_lower_prices  = prev_prices_wf * np.exp(wf_ci_lower_logrets)
    wf_ci_upper_prices  = prev_prices_wf * np.exp(wf_ci_upper_logrets)
    test_actual         = test_prices.values
    test_dates          = test_log_returns.index

    print(f"   - Walk-forward complete: {len(wf_predicted_prices)} next-day predictions with 95% CIs")

    # ══════════════════════════════════════════════════════════════════════
    #   SAVE PREDICTIONS TO CSV FOR CHART GENERATION
    #   Saves the walk-forward predictions, actual prices, and prediction
    #   intervals to a CSV file for use in the standalone chart script.
    # ══════════════════════════════════════════════════════════════════════
    pd.DataFrame({
        'Date':      test_dates,
        'Actual':    test_actual,
        'Predicted': wf_predicted_prices,
        'CI_Lower':  wf_ci_lower_prices,
        'CI_Upper':  wf_ci_upper_prices,
        'Model':     'ARIMA',
        'Ticker':    ticker
    }).to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_ARIMA_predictions.csv"), index=False)
    print(f"   - Saved: {ticker}_ARIMA_predictions.csv")

    print(f"\n[METRICS] Calculating performance metrics...")

    mape_oos        = mean_absolute_percentage_error(test_actual, wf_predicted_prices)
    rmse            = np.sqrt(mean_squared_error(test_actual, wf_predicted_prices))
    mae             = mean_absolute_error(test_actual, wf_predicted_prices)
    oos_rmse_logret = np.sqrt(mean_squared_error(test_log_returns.values, wf_predicted_logrets))

    actual_direction    = np.sign(np.diff(test_actual))
    predicted_direction = np.sign(np.diff(wf_predicted_prices))
    dir_accuracy        = np.mean(actual_direction == predicted_direction) * 100

    print(f"\n   Performance Metrics (Out-of-Sample, Next-Day Walk-Forward):")
    print(f"   {'─'*50}")
    print(f"   MAPE:                    {mape_oos:.4%}")
    print(f"   RMSE:                    {rmse:.4f}")
    print(f"   MAE:                     {mae:.4f}")
    print(f"   OOS RMSE (log-return):   {oos_rmse_logret:.6f}  ← scale-free, comparable across tickers")
    print(f"   Directional Accuracy:    {dir_accuracy:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    #   UNCERTAINTY METRICS — RAW AND NORMALISED WINKLER SCORE
    #
    #   Two Winkler scores are now reported:
    #     - Raw Winkler: in GBX price units, useful for within-ticker
    #       comparison across models (ARIMA vs RF vs GRU for same stock).
    #     - Normalised Winkler: raw score divided by mean actual price,
    #       producing a ratio that enables valid cross-ticker comparison 
    #       regardless of each stock's price level.
    #       e.g. HSBA raw 105 vs LLOY raw 7.6 is misleading; normalised
    #       scores account for HSBA trading approximately 13x higher than 
    #       LLOY.
    # ══════════════════════════════════════════════════════════════════════

    interval_widths    = wf_ci_upper_prices - wf_ci_lower_prices
    avg_interval_width = np.mean(interval_widths)
    coverage_rate      = np.mean((test_actual >= wf_ci_lower_prices) &
                                 (test_actual <= wf_ci_upper_prices)) * 100
    winkler_raw        = winkler_score(test_actual, wf_ci_lower_prices, wf_ci_upper_prices, alpha=0.05)
    winkler_norm       = winkler_score_normalised(test_actual, wf_ci_lower_prices, wf_ci_upper_prices, alpha=0.05)
    mean_actual_price  = np.mean(test_actual)

    print(f"\n   Uncertainty Metrics (95% CI — analytical, from model parameter uncertainty):")
    print(f"   NOTE: RF/GRU use empirical residual-std intervals; comparison is indicative only.")
    print(f"   {'─'*50}")
    print(f"   Avg Interval Width:         {avg_interval_width:.4f} GBX")
    print(f"   Coverage Rate:              {coverage_rate:.2f}%")
    print(f"   Mean Actual Price (test):   {mean_actual_price:.4f} GBX")
    print(f"   Winkler Score (95%, raw):   {winkler_raw:.4f}  ← GBX units, within-ticker comparison only")
    print(f"   Winkler Score (95%, norm):  {winkler_norm:.6f}  ← price-scaled, valid cross-ticker comparison")

    all_uncertainty_metrics.append({
        'Ticker': ticker, 'Model': 'ARIMA',
        'Avg_Interval_Width': avg_interval_width,
        'Coverage_Rate_%': coverage_rate,
        'Mean_Actual_Price': mean_actual_price,
        'Winkler_Score_Raw': winkler_raw,
        'Winkler_Score_Norm': winkler_norm,
        'Dir_Accuracy_%': dir_accuracy,
        'CI_Method': 'Analytical (ARIMA)'})

    if len(test_actual) > 1:
        arima_errors    = test_actual[1:] - wf_predicted_prices[1:]
        naive_errors    = test_actual[1:] - test_actual[:-1]
        dm_stat, dm_pval       = diebold_mariano_test(arima_errors, naive_errors)
        dm_significance        = "Significant" if dm_pval < 0.05 else "Not significant"
        print(f"\n   Diebold-Mariano Test (vs Naive Random Walk):")
        print(f"   {'─'*50}")
        print(f"   DM Statistic:            {dm_stat:.4f}")
        print(f"   p-value:                 {dm_pval:.4f}")
        print(f"   Result (α=0.05):         {dm_significance}")
        if dm_stat > 0:
            print(f"   Note: Positive DM stat → model errors larger than naive (RW outperforms)")
        else:
            print(f"   Note: Negative DM stat → model errors smaller than naive (model outperforms RW)")
        pd.DataFrame([{'Ticker': ticker, 'Model': f'ARIMA{best_order}',
                       'DM_Statistic': dm_stat, 'DM_pValue': dm_pval,
                       'Significant_005': dm_significance}]) \
          .to_csv(os.path.join(PER_TICKER_FOLDER,
              f"{ticker}_diebold_mariano.csv"), index=False)
        print(f"   - Saved: {ticker}_diebold_mariano.csv")

    print(f"\n[FORECAST] Generating 5-day forecast...")

    forecast_obj     = current_model.get_forecast(steps=5, alpha=0.05)
    forecast_logrets = np.asarray(forecast_obj.predicted_mean)

    ci_bounds_1step = current_model.get_forecast(steps=1, alpha=0.05).conf_int()
    ci_bounds_1step = np.asarray(ci_bounds_1step)
    sigma_1step     = (ci_bounds_1step[0, 1] - forecast_logrets[0]) / 1.96
    sigma_1step     = max(abs(sigma_1step), fallback_std)

    last_actual    = df[price_col].iloc[-1]
    last_date      = df.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

    horizons            = np.arange(1, 6)
    cumulative_log_rets = np.cumsum(forecast_logrets)
    forecast_prices     = last_actual * np.exp(cumulative_log_rets)
    ci_lower_prices     = last_actual * np.exp(cumulative_log_rets - 1.96 * sigma_1step * np.sqrt(horizons))
    ci_upper_prices     = last_actual * np.exp(cumulative_log_rets + 1.96 * sigma_1step * np.sqrt(horizons))


    # The weighted average gives more importance to the nearer days since
    # short-term forecasts from any model tend to be more reliable than
    # longer ones. Day 1 gets 50% of the weight, day 2 gets 20%, and the
    # remaining three days split the last 30% equally between them.
    day_weights     = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    weighted_avg    = np.sum(forecast_prices * day_weights)
    weighted_signal = "UP" if weighted_avg > last_actual else "DOWN"

    print(f"\n   Next 5 Trading Days Forecast:")
    print(f"   {'─'*80}")
    print(f"   Last Actual Price ({last_date.date()}): {last_actual:.3f} GBP")
    print(f"   Weighted Avg Forecast: {weighted_avg:.3f} GBP (50%-20%-10%-10%-10%)")
    print(f"   {'─'*80}")

    ticker_forecast_rows = []
    for i, (fdate, fprice, ci_low, ci_up, weight) in enumerate(zip(
            forecast_dates, forecast_prices, ci_lower_prices, ci_upper_prices, day_weights)):
        signal = "UP" if fprice > last_actual else "DOWN"
        print(f"   Day {i+1} ({fdate.date()}) [Weight: {weight:.0%}]:")
        print(f"      Forecast:  {fprice:.3f} GBP  {signal}")
        print(f"      95% CI:    [{ci_low:.3f}, {ci_up:.3f}] GBP")
        ticker_forecast_rows.append({
            'Ticker': ticker, 'Day': i+1, 'Date': fdate.date(),
            'Forecast_Price': fprice, 'Weight': weight,
            'Weighted_Contribution': fprice * weight,
            'CI_Lower': ci_low, 'CI_Upper': ci_up, 'Signal': signal})

    pd.DataFrame(ticker_forecast_rows).to_csv(
        os.path.join(PER_TICKER_FOLDER, f"{ticker}_5day_forecast.csv"), index=False)
    print(f"\n   - Saved: {ticker}_5day_forecast.csv")

    all_forecasts.append([
        ticker, f"{last_actual:.3f}",
        f"{forecast_prices[0]:.3f}", f"{forecast_prices[-1]:.3f}",
        f"{weighted_avg:.3f}", weighted_signal,
        f"[{ci_lower_prices[0]:.3f}, {ci_upper_prices[0]:.3f}]",
        f"[{ci_lower_prices[-1]:.3f}, {ci_upper_prices[-1]:.3f}]"])

    all_performance_rows.append({
        'Ticker': ticker, 'Model': 'ARIMA',
        'Order': str(best_order), 'Trend': best_trend,
        'Window':          best_window if best_window else 'Full',
        'Val_MAPE':        f"{best_config['val_mape']:.4%}",
        'OOS_MAPE':        f"{mape_oos:.4%}",
        'OOS_RMSE_LogRet': f"{oos_rmse_logret:.6f}",
        'RMSE':            f"{rmse:.4f}",
        'MAE':             f"{mae:.4f}",
        'Dir_Accuracy':    f"{dir_accuracy:.2f}%",
        'AIC':             f"{final_model.aic:.2f}",
        'Refit_Freq':      f"Every {WALK_REFIT_FREQ} days",
        'Test_Start':      test_dates[0].date(),
        'Test_End':        test_dates[-1].date(),
        'Test_Obs':        len(test_actual)})

    print(f"\n{'═'*100}")
    print(f"   - COMPLETED: {ticker}   ".center(100, "═"))
    print(f"{'═'*100}\n")


# ══════════════════════════════════════════════════════════════════════════
#   FINAL SUMMARY OUTPUTS
#   Once all three tickers have been processed, this section pulls the
#   results together into three summary tables and saves them as CSVs.
#   One for overall model performance, one for the interval quality
#   metrics, and one for the 5 day price forecasts. These are the files
#   I use directly in the dissertation for cross-model comparison.
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*100)
print("ARIMA MODEL - PERFORMANCE SUMMARY".center(100))
print("═"*100 + "\n")

performance_df = pd.DataFrame(all_performance_rows)
print(tabulate(performance_df, headers='keys', tablefmt='grid', showindex=False))
performance_df.to_csv(os.path.join(SUMMARY_FOLDER, "ARIMA_performance_summary.csv"), index=False)
print(f"\n- Saved: ARIMA_performance_summary.csv")

uncertainty_df = pd.DataFrame(all_uncertainty_metrics)
print("\nUncertainty Quantification Summary:")
print(tabulate(uncertainty_df, headers='keys', tablefmt='grid',
               showindex=False, floatfmt='.4f'))
uncertainty_df.to_csv(os.path.join(SUMMARY_FOLDER, "ARIMA_uncertainty_metrics.csv"), index=False)
print(f"\n- Saved: ARIMA_uncertainty_metrics.csv")

print("\n5-Day Price Forecasts with 95% CI:")
print(tabulate(all_forecasts,
               headers=['Ticker', 'Last Actual', 'Day 1', 'Day 5',
                        'Weighted Avg', 'Trend', 'Day 1 CI', 'Day 5 CI'],
               tablefmt='grid'))
forecast_df = pd.DataFrame(all_forecasts,
    columns=['Ticker', 'Last_Actual_Price', 'Day1_Forecast', 'Day5_Forecast',
             'Weighted_Avg', 'Trend', 'Day1_CI', 'Day5_CI'])
forecast_df.insert(1, 'Model', 'ARIMA')
forecast_df.to_csv(os.path.join(SUMMARY_FOLDER, "ARIMA_5day_forecasts_summary.csv"), index=False)
print(f"\n- Saved: ARIMA_5day_forecasts_summary.csv")

sys.stdout = _real_stdout
log_file.close()
