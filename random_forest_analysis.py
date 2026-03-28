# ══════════════════════════════════════════════════════════════════════════════════════════
#   Ross Todd
#   BSc (Hons) Software Development
#   Honours Project 2026 - Stock Market Prediction Comparison Analysis
#
#   Random Forest model — walk-forward validation and 5-day forecast for BARC, LLOY, HSBA
#
#   In this project, Random Forest is used as a multivariate model trained on
#   14 engineered technical features (lagged returns, SMAs, RSI, Bollinger band
#   width, volatility, volume ratio, ATR). ARIMA is a univariate baseline trained
#   on log-returns only, and GRU is a multivariate deep learning model trained on
#   10 features. ARIMA provides a classical statistical baseline against which the
#   advanced feature-based models can be benchmarked. All three share the same data
#   window, 80/20 split, walk-forward protocol, evaluation metrics, and refit
#   frequency (every 63 trading days) so that differences in predictive performance
#   are attributable to model class rather than evaluation methodology.
# ══════════════════════════════════════════════════════════════════════════════════════════

import pandas as pd
from data_loader import load_all_tickers
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from tabulate import tabulate
import joblib


# ══════════════════════════════════════════════════════════════════════════
#   OUTPUT FOLDERS
#   These are the folders where everything gets saved to. There is
#   one folder each for plots, per-ticker CSVs, and summary tables, all
#   sitting inside the main results folder. They get created automatically
#   if they don't exist yet so the script doesn't fail on a fresh run.
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_FOLDER     = "rf_results"
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
#   The three UK banking stocks I'm analysing, plus all the RF settings.
#   LAGS controls which previous days' returns get used as input features —
#   so lag 1 is yesterday's return, lag 2 is two days ago, and so on.
#   WALK_REFIT_FREQ controls how often the model gets retrained during
#   walk-forward — every 63 trading days, which is roughly one quarter.
# ══════════════════════════════════════════════════════════════════════════

TICKERS         = ['BARC.L', 'LLOY.L', 'HSBA.L']
LAGS            = [1, 2, 3, 5]
WALK_REFIT_FREQ = 63


# ══════════════════════════════════════════════════════════════════════════
#   HELPER FUNCTIONS
#   A few utility functions used throughout the script.
#   create_enhanced_features builds all 14 technical indicators from the
#   raw price and volume data — these are the inputs the RF model learns from.
#   get_prediction_intervals wraps the RF point forecast with a simple
#   95% interval based on the residual standard deviation from training.
#   diebold_mariano_test checks whether RF is actually doing better than
#   just guessing tomorrow will be the same as today (the naive baseline).
#   winkler_score measures how good the prediction intervals are — it
#   penalises both wide intervals and ones that miss the actual value.
#   winkler_score_normalised is the same thing but divided by the average
#   price, so I can fairly compare the three stocks against each other
#   even though they trade at very different price levels.
# ══════════════════════════════════════════════════════════════════════════

def create_enhanced_features(df, price_col='Adj Close'):
    df = df.copy()
    df['Log_Ret'] = np.log(df[price_col] / df[price_col].shift(1))
    for lag in LAGS:
        df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
    df['SMA_5']        = df[price_col].rolling(5).mean()
    df['SMA_20']       = df[price_col].rolling(20).mean()
    df['SMA_Ratio']    = df['SMA_5'] / df['SMA_20']
    delta              = df[price_col].diff()
    gain               = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss               = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs                 = gain / (loss + 1e-9)
    df['RSI']          = 100 - (100 / (1 + rs))
    sma                = df[price_col].rolling(20).mean()
    std                = df[price_col].rolling(20).std()
    df['BB_Width']     = ((sma + 2*std) - (sma - 2*std)) / (sma + 1e-9)
    df['Volatility']   = df['Log_Ret'].rolling(20).std()
    df['Volume_MA']    = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-9)
    df['True_Range']   = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df[price_col].shift(1)),
                   abs(df['Low']  - df[price_col].shift(1))))
    df['ATR_14'] = df['True_Range'].rolling(14).mean()
    df['ATR_20'] = df['True_Range'].rolling(20).mean()
    return df.dropna()


def get_prediction_intervals(rf_model, X, residual_std, horizon=1):
    point_pred  = rf_model.predict(X)
    scaled_std  = residual_std * np.sqrt(horizon)
    lower_bound = point_pred - 1.96 * scaled_std
    upper_bound = point_pred + 1.96 * scaled_std
    return point_pred, lower_bound, upper_bound


def diebold_mariano_test(errors1, errors2):
    errors1, errors2  = np.asarray(errors1), np.asarray(errors2)
    loss_differential = errors1**2 - errors2**2
    dm_stat  = np.mean(loss_differential) / np.sqrt(np.var(loss_differential, ddof=1) / len(loss_differential))
    p_value  = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def winkler_score(actual, lower, upper, alpha=0.05):
    # Raw Winkler score in price units (GBX). Use winkler_score_normalised
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
    #   This allows valid cross-ticker comparison between BARC (452p),
    #   LLOY (102p), and HSBA (1393p), which the raw score cannot support
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
#   loads the data, builds the technical features, does the train/test
#   split, runs a randomised grid search to find the best RF settings,
#   then does walk-forward validation to get realistic out-of-sample
#   predictions. After that it calculates all the performance metrics,
#   generates the plots, and produces the 5 day forecast. Results are
#   saved to CSV as it goes.
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*100)
print("RANDOM FOREST ANALYSIS - UK BANKING STOCKS (NEXT-DAY WALK-FORWARD + 5-DAY FORECAST)".center(100))
print("═"*100)
print(f"  Data window : {DATA_START}  →  {DATA_END}  (5 calendar years, 80/20 split)")
print(f"  Split       : 80% train / 20% test")
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

    print(f"[DATA PREVIEW] Data loaded for {ticker}...")
    raw_data = ticker_data[ticker]

    available_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                      if c in raw_data.columns]
    inspect_df = raw_data[available_cols].copy()
    inspect_df.index = inspect_df.index.date
    snapshot = pd.concat([inspect_df.head(5), inspect_df.tail(5)])
    print(f"\n   Data Preview (first 5 / last 5 rows):")
    print(tabulate(snapshot.reset_index(), headers=['Date'] + list(snapshot.columns),
                   tablefmt='simple', floatfmt='.2f'))
    print(f"   Total observations: {len(raw_data)}")
    print(f"   Date range: {raw_data.index[0].date()} → {raw_data.index[-1].date()}")

    print(f"\n[FEATURES] Creating technical features and applying train/test split...")

    price_col       = 'Adj Close' if 'Adj Close' in raw_data.columns else 'Close'
    df_feat         = create_enhanced_features(raw_data, price_col=price_col)
    feature_columns = [c for c in df_feat.columns if c.startswith((
        'Ret_Lag', 'SMA', 'RSI', 'BB', 'Volatility', 'Volume', 'ATR'))]

    print(f"   Features created: {len(feature_columns)}")
    print(f"   Feature list: {', '.join(feature_columns[:5])}...")

    X = df_feat[feature_columns].iloc[:-1].copy()
    y = df_feat['Log_Ret'].shift(-1).iloc[:-1].copy()

    if len(X) < 10:
        print(f"   ✗ Insufficient data for {ticker}"); continue

    split_idx = int(len(X) * TRAIN_RATIO)

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    all_prices_arr     = df_feat[price_col].values
    all_dates_arr      = df_feat.index
    test_actual_prices = all_prices_arr[split_idx : split_idx + len(X_test)]
    prev_prices        = all_prices_arr[split_idx - 1 : split_idx - 1 + len(X_test)]
    test_price_dates   = all_dates_arr[split_idx : split_idx + len(X_test)]

    print(f"   Train : {len(X_train)} obs  "
          f"({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Test  : {len(X_test)} obs  "
          f"({X_test.index[0].date()} → {X_test.index[-1].date()})")
    print(f"   Ratio : {len(X_train)/(len(X_train)+len(X_test)):.4f} / "
          f"{len(X_test)/(len(X_train)+len(X_test)):.4f}")

    print(f"\n[SCALING] Scaling features...")
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"   - Features standardised (mean=0, std=1) using training-set statistics only")
    ticker_clean = ticker.replace(".", "_")
    joblib.dump(scaler, f'saved_models/scaler_{ticker_clean}_rf.pkl')
    print(f'   - Saved scaler: scaler_{ticker_clean}_rf.pkl')

    print(f"\n[GRID SEARCH] Running randomised search...")
    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators':      [300, 800, 1200, 2000, 3000],
        'max_depth':         [3, 5, 8, 12, 20, None],
        'min_samples_split': [2, 5, 10, 20, 40],
        'min_samples_leaf':  [1, 2, 5, 10, 20],
        'max_features':      ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap':         [True],
        'max_samples':       [0.5, 0.7, 0.9],
        'ccp_alpha':         [0.0, 0.0001, 0.001, 0.01]
    }

    print(f"   Testing 50 random combinations (bootstrap=True)...")
    grid_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        param_dist,
        n_iter=50, cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42, n_jobs=1, verbose=0)
    grid_search.fit(X_train_scaled, y_train)
    print(f"   - Search complete")

    # ══════════════════════════════════════════════════════════════════════
    #   GRID SEARCH RESULTS
    #   Val_RMSE_LogRet is derived directly from mean_test_score by
    #   converting the negative MSE back to RMSE. This gives each of the
    #   50 configurations its own independently calculated validation RMSE
    #   on log-returns, so the top 3 rows can be used directly in the
    #   dissertation table without needing a separate best_hyperparameters
    #   file. bootstrap is excluded from keep_cols as it is fixed to True
    #   for all configurations and adds no information.
    # ══════════════════════════════════════════════════════════════════════

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    keep_cols     = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
                     'param_min_samples_leaf', 'param_max_features',
                     'param_max_samples', 'param_ccp_alpha', 'mean_test_score', 'rank_test_score']
    cv_results_df = cv_results_df[keep_cols].sort_values('rank_test_score')
    cv_results_df.insert(0, 'Ticker', ticker)
    cv_results_df['Val_RMSE_LogRet'] = np.sqrt(-cv_results_df['mean_test_score'])
    cv_results_df.to_csv(os.path.join(PER_TICKER_FOLDER,
        f"{ticker}_grid_search_rf.csv"), index=False)
    print(f"   - Saved: {ticker}_grid_search_rf.csv")

    print(f"\n   Top 5 Configurations by Validation Score:")
    display_cols = ['param_n_estimators', 'param_max_depth', 'mean_test_score', 'rank_test_score']
    print(tabulate(cv_results_df.head(5)[display_cols], headers='keys',
                   tablefmt='simple', showindex=False, floatfmt='.6f'))

    best_rf     = grid_search.best_estimator_
    best_params = grid_search.best_params_

    val_split_idx    = int(len(X_train_scaled) * 0.8)
    val_preds_logret = best_rf.predict(X_train_scaled[val_split_idx:])
    val_true_logret  = y_train.values[val_split_idx:]
    val_rmse_logret  = np.sqrt(mean_squared_error(val_true_logret, val_preds_logret))

    print(f"\n   Best Configuration:")
    print(f"   Validation RMSE (log-return): {val_rmse_logret:.6f}")

    print(f"\n[WALK-FORWARD] Running walk-forward validation "
          f"(refit every {WALK_REFIT_FREQ} days)...")

    val_resid_preds  = best_rf.predict(X_train_scaled[val_split_idx:])
    val_residuals    = y_train.values[val_split_idx:] - val_resid_preds
    residual_std     = max(np.std(val_residuals, ddof=1), 1e-6)
    print(f"   Initial residual std (log-return space): {residual_std:.6f}")

    wf_predicted_logrets = []
    wf_ci_lower_logrets  = []
    wf_ci_upper_logrets  = []
    current_model        = best_rf
    current_residual_std = residual_std

    for step_idx in range(len(X_test_scaled)):

        if step_idx > 0 and step_idx % WALK_REFIT_FREQ == 0:
            X_expanded = np.vstack([X_train_scaled, X_test_scaled[:step_idx]])
            y_expanded = np.concatenate([y_train.values, y_test.iloc[:step_idx].values])
            current_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=1)
            current_model.fit(X_expanded, y_expanded)
            refit_val_start      = int(len(X_expanded) * 0.8)
            current_residual_std = max(np.std(
                y_expanded[refit_val_start:] - current_model.predict(X_expanded[refit_val_start:]),
                ddof=1), 1e-6)

        predicted_logret, lower_logret, upper_logret = get_prediction_intervals(
            current_model, X_test_scaled[step_idx].reshape(1, -1), current_residual_std, horizon=1)

        wf_predicted_logrets.append(predicted_logret[0])
        wf_ci_lower_logrets.append(lower_logret[0])
        wf_ci_upper_logrets.append(upper_logret[0])

    wf_predicted_prices = prev_prices * np.exp(wf_predicted_logrets)
    wf_ci_lower_prices  = prev_prices * np.exp(wf_ci_lower_logrets)
    wf_ci_upper_prices  = prev_prices * np.exp(wf_ci_upper_logrets)
    print(f"   - Walk-forward complete: {len(wf_predicted_prices)} predictions "
          f"(refit every {WALK_REFIT_FREQ} days)")

    test_actual = test_actual_prices
    test_dates  = test_price_dates

    print(f"\n[METRICS] Calculating performance metrics...")

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
        'Model':     'RF',
        'Ticker':    ticker
    }).to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_RF_predictions.csv"), index=False)
    print(f"   - Saved: {ticker}_RF_predictions.csv")

    mape_oos        = mean_absolute_percentage_error(test_actual, wf_predicted_prices)
    rmse            = np.sqrt(mean_squared_error(test_actual, wf_predicted_prices))
    mae             = mean_absolute_error(test_actual, wf_predicted_prices)
    oos_rmse_logret = np.sqrt(mean_squared_error(y_test.values, wf_predicted_logrets))

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

    if len(test_actual) > 1:
        rf_errors       = test_actual[1:] - wf_predicted_prices[1:]
        naive_errors    = test_actual[1:] - test_actual[:-1]
        dm_stat, dm_pval       = diebold_mariano_test(rf_errors, naive_errors)
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
        pd.DataFrame([{'Ticker': ticker, 'Model': f'RF_refit{WALK_REFIT_FREQ}d',
                       'DM_Statistic': dm_stat, 'DM_pValue': dm_pval,
                       'Significant_005': dm_significance}]) \
          .to_csv(os.path.join(PER_TICKER_FOLDER,
              f"{ticker}_diebold_mariano.csv"), index=False)
        print(f"   - Saved: {ticker}_diebold_mariano.csv")

    # ══════════════════════════════════════════════════════════════════════
    #   UNCERTAINTY METRICS — RAW AND NORMALISED WINKLER SCORE
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[UNCERTAINTY] Calculating uncertainty metrics...")

    interval_widths    = wf_ci_upper_prices - wf_ci_lower_prices
    avg_interval_width = np.mean(interval_widths)
    coverage_rate      = np.mean((test_actual >= wf_ci_lower_prices) &
                                 (test_actual <= wf_ci_upper_prices)) * 100
    winkler_raw        = winkler_score(test_actual, wf_ci_lower_prices, wf_ci_upper_prices, alpha=0.05)
    winkler_norm       = winkler_score_normalised(test_actual, wf_ci_lower_prices, wf_ci_upper_prices, alpha=0.05)
    mean_actual_price  = np.mean(test_actual)

    print(f"\n   Uncertainty Metrics (95% PI — empirical residual-std-based):")
    print(f"   NOTE: ARIMA uses analytical CIs; comparison is indicative only.")
    print(f"   {'─'*50}")
    print(f"   Avg Interval Width:         {avg_interval_width:.4f} GBX")
    print(f"   Coverage Rate:              {coverage_rate:.2f}%")
    print(f"   Mean Actual Price (test):   {mean_actual_price:.4f} GBX")
    print(f"   Winkler Score (95%, raw):   {winkler_raw:.4f}  ← GBX units, within-ticker comparison only")
    print(f"   Winkler Score (95%, norm):  {winkler_norm:.6f}  ← price-scaled, valid cross-ticker comparison")

    all_uncertainty_metrics.append({
        'Ticker': ticker, 'Model': 'RF',
        'Avg_Interval_Width': avg_interval_width,
        'Coverage_Rate_%': coverage_rate,
        'Mean_Actual_Price': mean_actual_price,
        'Winkler_Score_Raw': winkler_raw,
        'Winkler_Score_Norm': winkler_norm,
        'Dir_Accuracy_%': dir_accuracy,
        'PI_Method': 'Empirical (residual std)'})

    print(f"\n[FORECAST] Generating 5-day forecast...")

    pd.DataFrame({'Feature': feature_columns, 'Importance': best_rf.feature_importances_}) \
      .sort_values('Importance', ascending=False) \
      .assign(Ticker=ticker, Model='RF') \
      .to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_feature_importance.csv"), index=False)
    print(f"   - Saved: {ticker}_feature_importance.csv")

    scaler               = joblib.load(f'saved_models/scaler_{ticker_clean}_rf.pkl')
    current_residual_std = max(np.std(
        y_train.values[val_split_idx:] - best_rf.predict(X_train_scaled[val_split_idx:]),
        ddof=1), 1e-6)
    current_model        = best_rf

    last_feature_row  = df_feat[feature_columns].iloc[-1].copy()
    last_price        = df_feat[price_col].iloc[-1]
    last_date         = df_feat.index[-1]
    forecast_dates    = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
    day_weights       = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    forecast_log_rets    = []
    current_features     = last_feature_row.copy()
    current_price        = last_price

    running_rsi_approx   = df_feat['RSI'].iloc[-1]
    running_sq_ret       = current_features.get('Volatility', 0) ** 2
    running_true_range   = current_features.get('ATR_14', 0)

    for day in range(5):
        current_features_scaled = scaler.transform(current_features.values.reshape(1, -1))
        pred_logret, _, _ = get_prediction_intervals(
            current_model, current_features_scaled, current_residual_std, horizon=1)
        forecast_log_rets.append(pred_logret[0])

        pred_price = current_price * np.exp(pred_logret[0])

        for lag in reversed(LAGS[1:]):
            if f'Ret_Lag_{lag}' in current_features.index:
                prev_lag = f'Ret_Lag_{lag-1}' if lag > 1 else None
                current_features[f'Ret_Lag_{lag}'] = (
                    current_features[prev_lag] if prev_lag and prev_lag in current_features.index
                    else 0)
        if 'Ret_Lag_1' in current_features.index:
            current_features['Ret_Lag_1'] = pred_logret[0]

        if 'SMA_5' in current_features.index:
            current_features['SMA_5']  = (current_features['SMA_5'] * 4 + pred_price) / 5
        if 'SMA_20' in current_features.index:
            current_features['SMA_20'] = (current_features['SMA_20'] * 19 + pred_price) / 20
        if 'SMA_Ratio' in current_features.index:
            current_features['SMA_Ratio'] = current_features['SMA_5'] / (
                current_features['SMA_20'] + 1e-9)

        if 'Volatility' in current_features.index:
            alpha_vol      = 1.0 / 20
            running_sq_ret = (1 - alpha_vol) * running_sq_ret + alpha_vol * pred_logret[0]**2
            current_features['Volatility'] = np.sqrt(max(running_sq_ret, 1e-10))

        if 'BB_Width' in current_features.index:
            ewm_std = current_features['Volatility'] * current_features['SMA_20']
            current_features['BB_Width'] = (4 * ewm_std) / (current_features['SMA_20'] + 1e-9)

        if 'ATR_14' in current_features.index:
            daily_range_proxy  = pred_price * current_features['Volatility']
            running_true_range = (running_true_range * 13 + daily_range_proxy) / 14
            current_features['ATR_14'] = running_true_range
        if 'ATR_20' in current_features.index:
            current_features['ATR_20'] = (current_features['ATR_20'] * 19 + running_true_range) / 20

        if 'RSI' in current_features.index:
            alpha_rsi = 1.0 / 14
            if pred_logret[0] > 0:
                running_rsi_approx = (1 - alpha_rsi) * current_features['RSI'] + alpha_rsi * 100
            else:
                running_rsi_approx = (1 - alpha_rsi) * current_features['RSI']
            current_features['RSI'] = np.clip(running_rsi_approx, 0, 100)

        current_price = pred_price

    forecast_log_rets   = np.array(forecast_log_rets)
    cumulative_log_rets = np.cumsum(forecast_log_rets)
    horizons            = np.arange(1, 6)
    forecast_prices     = last_price * np.exp(cumulative_log_rets)
    forecast_ci_lower   = last_price * np.exp(cumulative_log_rets - 1.96 * current_residual_std * np.sqrt(horizons))
    forecast_ci_upper   = last_price * np.exp(cumulative_log_rets + 1.96 * current_residual_std * np.sqrt(horizons))

    # The weighted average gives more importance to the nearer days since
    # short-term forecasts from any model tend to be more reliable than
    # longer ones. Day 1 gets 50% of the weight, day 2 gets 20%, and the
    # remaining three days split the last 30% equally between them.
    weighted_avg        = np.sum(forecast_prices * day_weights)
    weighted_signal     = "UP" if weighted_avg > last_price else "DOWN"

    print(f"\n   Next 5 Trading Days Forecast (Recursive, with 95% PI):")
    print(f"   {'─'*80}")
    print(f"   Last Actual Price ({last_date.date()}): {last_price:.3f} GBP")
    print(f"   Weighted Avg Forecast: {weighted_avg:.3f} GBP (50%-20%-10%-10%-10%)")
    print(f"   {'─'*80}")

    ticker_forecast_rows = []
    for i, (fdate, fprice, ci_low, ci_up, weight) in enumerate(zip(
            forecast_dates, forecast_prices, forecast_ci_lower, forecast_ci_upper, day_weights)):
        signal = "UP" if fprice > last_price else "DOWN"
        print(f"   Day {i+1} ({fdate.date()}) [Weight: {weight:.0%}]:")
        print(f"      Forecast:  {fprice:.3f} GBP  {signal}")
        print(f"      95% PI:    [{ci_low:.3f}, {ci_up:.3f}] GBP")
        ticker_forecast_rows.append({
            'Ticker': ticker, 'Day': i+1, 'Date': fdate.date(),
            'Forecast_Price': fprice, 'Weight': weight,
            'Weighted_Contribution': fprice * weight,
            'PI_Lower': ci_low, 'PI_Upper': ci_up, 'Signal': signal})

    pd.DataFrame(ticker_forecast_rows).to_csv(
        os.path.join(PER_TICKER_FOLDER, f"{ticker}_5day_forecast.csv"), index=False)
    print(f"\n   - Saved: {ticker}_5day_forecast.csv")

    all_forecasts.append([
        ticker, f"{last_price:.3f}",
        f"{forecast_prices[0]:.3f}", f"{forecast_prices[-1]:.3f}",
        f"{weighted_avg:.3f}", weighted_signal,
        f"[{forecast_ci_lower[0]:.3f}, {forecast_ci_upper[0]:.3f}]",
        f"[{forecast_ci_lower[-1]:.3f}, {forecast_ci_upper[-1]:.3f}]"])

    config_str = (f"n_est={best_params['n_estimators']}|"
                  f"depth={best_params['max_depth']}|"
                  f"refit={WALK_REFIT_FREQ}d")
    all_performance_rows.append({
        'Ticker':            ticker,
        'Model':             'RF',
        'Val_RMSE_LogRet':   f"{val_rmse_logret:.6f}",
        'OOS_RMSE_LogRet':   f"{oos_rmse_logret:.6f}",
        'OOS_MAPE':          f"{mape_oos:.4%}",
        'RMSE':              f"{rmse:.4f}",
        'MAE':               f"{mae:.4f}",
        'Dir_Accuracy':      f"{dir_accuracy:.2f}%",
        'Best_Config':       config_str,
        'Refit_Freq':        f"Every {WALK_REFIT_FREQ} days",
        'Test_Start':        test_dates[0].date(),
        'Test_End':          test_dates[-1].date(),
        'Test_Obs':          len(test_actual)})

    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(best_rf, f'saved_models/rf_{ticker.replace(".", "_")}.pkl')
    print(f"   - Saved model: rf_{ticker.replace('.', '_')}.pkl")

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
print("RANDOM FOREST MODEL - PERFORMANCE SUMMARY".center(100))
print("═"*100 + "\n")

performance_df = pd.DataFrame(all_performance_rows)
print(tabulate(performance_df, headers='keys', tablefmt='grid', showindex=False))
performance_df.to_csv(os.path.join(SUMMARY_FOLDER, "RF_performance_summary.csv"), index=False)
print(f"\n- Saved: RF_performance_summary.csv")

uncertainty_df = pd.DataFrame(all_uncertainty_metrics)
print("\nUncertainty Quantification Summary:")
print(tabulate(uncertainty_df, headers='keys', tablefmt='grid',
               showindex=False, floatfmt='.4f'))
uncertainty_df.to_csv(os.path.join(SUMMARY_FOLDER, "RF_uncertainty_metrics.csv"), index=False)
print(f"\n- Saved: RF_uncertainty_metrics.csv")

print("\n5-Day Price Forecasts with 95% PI:")
print(tabulate(all_forecasts,
               headers=['Ticker', 'Last Actual', 'Day 1', 'Day 5',
                        'Weighted Avg', 'Trend', 'Day 1 PI', 'Day 5 PI'],
               tablefmt='grid'))
forecast_df = pd.DataFrame(all_forecasts,
    columns=['Ticker', 'Last_Actual_Price', 'Day1_Forecast', 'Day5_Forecast',
             'Weighted_Avg', 'Trend', 'Day1_PI', 'Day5_PI'])
forecast_df.insert(1, 'Model', 'RF')
forecast_df.to_csv(os.path.join(SUMMARY_FOLDER, "RF_5day_forecasts_summary.csv"), index=False)
print(f"\n- Saved: RF_5day_forecasts_summary.csv")

sys.stdout = _real_stdout
log_file.close()
