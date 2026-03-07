# ═════════════════════════════════════════════════════════════════════════════════
#   Ross Todd
#   BSc (Hons) Software Development
#   Honours Project 2026 - Stock Market Prediction Comparison Analysis
#
#   GRU model — walk-forward validation and 5-day forecast for BARC, LLOY, HSBA
#
#   In this project, GRU is used as a multivariate deep learning model trained on
#   10 engineered features (OHLCV, RSI, MACD, Bollinger bands, lagged log-return).
#   ARIMA is a univariate baseline trained on log-returns only, and Random Forest
#   is a multivariate model trained on 14 features. ARIMA provides a classical
#   statistical baseline against which the advanced feature-based models can be
#   benchmarked. All three share the same data window, 80/20 split, walk-forward
#   protocol, evaluation metrics, and refit frequency (every 63 trading days) so
#   that differences in predictive performance are attributable to model class
#   rather than evaluation methodology.
# ═════════════════════════════════════════════════════════════════════════════════

import sys
import warnings
import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error, mean_absolute_error)
from scipy.stats import norm
from tabulate import tabulate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump as joblib_dump
from statsmodels.graphics.tsaplots import plot_acf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_all_tickers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# ══════════════════════════════════════════════════════════════════════════
#   OUTPUT FOLDERS
#   These are the folders where everything gets saved to. There is
#   one folder each for plots, per-ticker CSVs, and summary tables, all
#   sitting inside the main results folder. They get created automatically
#   if they don't exist yet so the script doesn't fail on a fresh run.
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_FOLDER     = "gru_results"
PLOTS_FOLDER      = os.path.join(OUTPUT_FOLDER, "plots")
PER_TICKER_FOLDER = os.path.join(OUTPUT_FOLDER, "per_ticker_results")
SUMMARY_FOLDER    = os.path.join(OUTPUT_FOLDER, "summary")

for folder in [OUTPUT_FOLDER, PLOTS_FOLDER, PER_TICKER_FOLDER, SUMMARY_FOLDER]:
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
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

LOG_FILENAME = "terminal_output.txt"
log_file     = open(os.path.join(OUTPUT_FOLDER, LOG_FILENAME), mode="w", encoding="utf-8")
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
#   The three UK banking stocks I'm analysing, plus all the GRU settings.
#   param_grid defines the search space for the randomised hyperparameter
#   search — lookback length, number of GRU units, layers, dropout rate,
#   learning rate, and batch size. WALK_REFIT_FREQ is fixed at 63 trading
#   days to match the ARIMA and RF refit schedules so the walk-forward
#   comparison is fair. get_units_for_lookback ties network width to
#   sequence length to stop the search trying combinations that are either
#   too small or too large for the amount of history being fed in.
# ══════════════════════════════════════════════════════════════════════════

TICKERS         = ['BARC.L', 'LLOY.L', 'HSBA.L']
WALK_REFIT_FREQ = 63

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

param_grid = {
    'lookback':   [21, 63, 126],
    'units':      [32, 64, 96, 128],
    'layers':     [1, 2],
    'dropout':    [0.2, 0.3, 0.5],
    'lr':         [0.001, 0.0005],
    'epochs':     [50],
    'batch_size': [32, 64]
}


# ══════════════════════════════════════════════════════════════════════════
#   HELPER FUNCTIONS
#   A few utility functions used throughout the script.
#   create_sequences turns the scaled data into overlapping input/output
#   windows of length lookback — this is the format the GRU expects.
#   build_gru_model constructs the network with the given hyperparameters.
#   get_prediction_intervals wraps the GRU point forecast with a 95%
#   interval based on the residual standard deviation from training.
#   get_units_for_lookback ties network width to sequence length so the
#   grid search doesn't try combinations that are too big or too small.
#   diebold_mariano_test checks whether GRU is actually doing better than
#   just guessing tomorrow will be the same as today (the naive baseline).
#   winkler_score measures how good the prediction intervals are — it
#   penalises both wide intervals and ones that miss the actual value.
#   winkler_score_normalised is the same thing but divided by the average
#   price, so I can fairly compare the three stocks against each other
#   even though they trade at very different price levels.
# ══════════════════════════════════════════════════════════════════════════

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, -1])
    return np.array(X), np.array(y)


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


def build_gru_model(lookback, n_features, units=64, layers=1, dropout=0.2, lr=0.001):
    model = Sequential()
    if layers == 1:
        model.add(GRU(units, input_shape=(lookback, n_features)))
    else:
        model.add(GRU(units, return_sequences=True, input_shape=(lookback, n_features)))
        for _ in range(layers - 2):
            model.add(GRU(units, return_sequences=True))
        model.add(GRU(units))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='mse', metrics=['mae'])
    return model


def get_prediction_intervals(keras_model, X_new, residual_std, horizon=1):
    # residual_std is in scaled space; intervals widen with sqrt(horizon)
    # 1.96 corresponds to a 95% prediction interval
    point_pred  = keras_model.predict(X_new, verbose=0).flatten()
    scaled_std  = residual_std * np.sqrt(horizon)
    lower_bound = point_pred - 1.96 * scaled_std
    upper_bound = point_pred + 1.96 * scaled_std
    return point_pred, lower_bound, upper_bound


def get_units_for_lookback(lb):
    # Ties network width to sequence length to keep search tractable.
    # Longer sequences have more temporal context so benefit from wider networks.
    if lb <= 40:
        return [32, 64]
    elif lb <= 120:
        return [64, 96]
    else:
        return [96, 128]


# ══════════════════════════════════════════════════════════════════════════
#   DIAGNOSTIC PLOT FUNCTIONS
#   These four functions handle all the chart saving for each ticker.
#   create_prediction_plot_with_ci makes a four panel figure — actual vs
#   predicted with the interval band, a scatter plot, errors over time,
#   and a histogram of the errors. Good for spotting any obvious issues.
#   create_residual_acf_plot checks whether the model residuals still have
#   any pattern left in them — ideally they shouldn't.
#   create_training_history_plot shows the training and validation loss
#   and MAE curves across epochs, useful for checking for overfitting.
#   create_feature_importance_plot shows which of the 10 input features
#   the GRU relied on most, based on permutation importance.
#   create_forecast_chart shows the last 30 trading days of real prices
#   joined up to the 5 day forecast, with the interval band shaded in.
# ══════════════════════════════════════════════════════════════════════════

def create_prediction_plot_with_ci(actual, predicted, ci_lower, ci_upper,
                                   ticker, output_folder, dates):
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(dates, actual, label='Actual', linewidth=1.5, alpha=0.8, color='blue')
    ax1.plot(dates, predicted, label='Predicted', linewidth=1.5, alpha=0.8, color='red')
    ax1.fill_between(dates, ci_lower, ci_upper, alpha=0.2, color='red', label='95% PI (residual-based)')
    ax1.set_title(f'{ticker} - Actual vs Predicted Prices with 95% PI (GRU)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Price (GBP)')
    ax1.legend(loc='best'); ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(actual, predicted, alpha=0.5, s=20)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_title(f'{ticker} - Prediction Scatter (GRU)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Actual Price (GBP)'); ax2.set_ylabel('Predicted Price (GBP)')
    ax2.legend(loc='best'); ax2.grid(True, alpha=0.3)

    errors = actual - predicted
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(dates, errors, linewidth=0.8, color='darkred')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title(f'{ticker} - Forecast Errors Over Time (GRU)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date'); ax3.set_ylabel('Error (Actual - Predicted)')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_title(f'{ticker} - Error Distribution (GRU)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Prediction Error (GBP)'); ax4.set_ylabel('Frequency')
    ax4.legend(loc='best'); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{ticker}_GRU_predictions.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   - Saved: {filename}")


def create_residual_acf_plot(residuals, ticker, output_folder):
    if len(residuals) < 10:
        print(f"    Not enough residuals for ACF plot"); return
    fig, ax = plt.subplots(figsize=(12, 5))
    max_lags = min(40, len(residuals) // 2)
    plot_acf(residuals, lags=max_lags, ax=ax, alpha=0.05)
    ax.set_title(f'{ticker} - Residual Autocorrelation (GRU)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Lag'); ax.set_ylabel('Autocorrelation'); ax.grid(True, alpha=0.3)
    textstr = 'Residuals should show no significant\nautocorrelation (stay within blue bands)\nif model captures temporal dependencies.'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.tight_layout()
    filename = f"{ticker}_GRU_residual_acf.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   - Saved: {filename}")


def create_training_history_plot(history, ticker, config_str, output_folder):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'],     label='Training Loss', linewidth=1.5)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=1.5)
    axes[0].set_title(f'{ticker} - Training Loss (GRU)\n{config_str}', fontweight='bold', fontsize=11)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history['mae'],     label='Training MAE', linewidth=1.5)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=1.5)
    axes[1].set_title(f'{ticker} - Training MAE (GRU)\n{config_str}', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{ticker}_GRU_training_history.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   - Saved: {filename}")


def create_feature_importance_plot(importances, feature_names, ticker, output_folder):
    if len(importances) == 0 or len(feature_names) == 0:
        print(f"    No features for importance plot"); return
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(15, len(importances))
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[sorted_idx[:top_n]], color='cornflowerblue', edgecolor='navy')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (ΔMAPE when permuted)', fontsize=11)
    ax.set_title(f'{ticker} - GRU Feature Importance (Top {top_n})\n'
                 f'(Permutation-based: cross-sample shuffling within feature column)',
                 fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(importances[sorted_idx[:top_n]]):
        ax.text(v + 0.0001, i, f"{v:.6f}", va='center', fontsize=9)
    plt.tight_layout()
    filename = f"{ticker}_GRU_feature_importance.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   - Saved: {filename}")


def create_forecast_chart(ticker, df, price_col, last_actual, last_date,
                          forecast_dates, forecast_prices,
                          ci_lower_prices, ci_upper_prices,
                          model_label, output_folder, n_history=30):
    """
    Saves a line chart showing the last n_history trading days of actual
    prices followed by the 5-day forward forecast with 95% CI/PI ribbon.
    model_label is used in the title and filename (e.g. 'ARIMA(1,0,1)', 'GRU', 'RF').
    """
    hist_prices = df[price_col].iloc[-n_history:]
    hist_dates  = hist_prices.index

    # Stitch history endpoint to forecast so the line is continuous
    bridge_dates  = [last_date]   + list(forecast_dates)
    bridge_prices = [last_actual] + list(forecast_prices)
    bridge_lower  = [last_actual] + list(ci_lower_prices)
    bridge_upper  = [last_actual] + list(ci_upper_prices)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.axvspan(bridge_dates[0], bridge_dates[-1],
               alpha=0.06, color='gold', label='Forecast window')

    ax.plot(hist_dates, hist_prices.values,
            color='#1f77b4', linewidth=2.0, label='Actual (last 30 days)')
    ax.plot(hist_dates, hist_prices.values,
            'o', color='#1f77b4', markersize=3, alpha=0.6)

    ax.axvline(x=last_date, color='grey', linestyle='--',
               linewidth=1.2, alpha=0.7, label=f'Last actual ({last_date.date()})')

    ax.fill_between(bridge_dates, bridge_lower, bridge_upper,
                    color='#ff7f0e', alpha=0.18, label='95% CI/PI')

    ax.plot(bridge_dates, bridge_prices,
            color='#ff7f0e', linewidth=2.2,
            linestyle='--', marker='o', markersize=6,
            label='Forecast')

    ax.set_title(f'{ticker}  –  Last 30 Trading Days + 5-Day Forecast  ({model_label})',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price (GBX)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    filename = f"{ticker}_GRU_30day_history_5day_forecast.png"
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   - Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════
#   MAIN ANALYSIS LOOP
#   This is where everything actually runs. For each ticker, the script
#   loads the data, builds the technical features, scales everything, runs
#   a randomised grid search to find the best GRU architecture, then does
#   walk-forward validation to get realistic out-of-sample predictions.
#   After that it calculates all the performance metrics, runs permutation
#   feature importance, generates the plots, and produces the 5 day
#   forecast. Results are saved to CSV as it goes.
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 100)
print("GRU ANALYSIS - UK BANKING STOCKS (NEXT-DAY WALK-FORWARD + 5-DAY FORECAST)".center(100))
print("═" * 100)
print(f"  Data window : {DATA_START}  →  {DATA_END}  (5 calendar years, 80/20 split)")
print(f"  Split       : 80% train / 20% test")
print(f"  Walk-forward refit frequency: every {WALK_REFIT_FREQ} trading days (1 quarter)")
print("═" * 100 + "\n")
print("[DATA] Downloading all tickers via data_loader...\n")
ticker_data = load_all_tickers(TICKERS, DATA_START, DATA_END, delay=10, verbose=True)
print()

all_performance_rows    = []
all_forecasts           = []
all_uncertainty_metrics = []

for ticker in TICKERS:
    if ticker not in ticker_data:
        print(f" Skipping {ticker} – download failed.")
        continue

    print("\n" + "═" * 100)
    print(f"   PROCESSING: {ticker}   ".center(100, "═"))
    print("═" * 100 + "\n")

    raw_data = ticker_data[ticker]

    print(f"[DATA PREVIEW] Data loaded for {ticker}...")
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

    print(f"\n[FEATURES] Preprocessing and creating features...")
    df = raw_data[['Adj Close', 'Open', 'High', 'Low', 'Volume']].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().astype(float)

    delta = df['Adj Close'].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI']         = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df['MACD']        = df['Adj Close'].ewm(span=12).mean() - df['Adj Close'].ewm(span=26).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['MA20']        = df['Adj Close'].rolling(window=20).mean()
    df['StdDev']      = df['Adj Close'].rolling(window=20).std()
    df['Upper_Band']  = df['MA20'] + (df['StdDev'] * 2)
    df['Lower_Band']  = df['MA20'] - (df['StdDev'] * 2)
    df = df.dropna()

    prices      = df['Adj Close'].values
    log_returns = np.log(prices[1:] / prices[:-1])
    df          = df.iloc[1:].copy()
    df['log_return']   = log_returns
    df['Log_Ret_Lag1'] = df['log_return'].shift(1)
    df = df.dropna()

    feature_columns = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD',
                       'Signal_Line', 'Upper_Band', 'Lower_Band', 'Log_Ret_Lag1']
    # n_features includes the target column 'log_return' which is the last column
    # in the scaled data array alongside the 10 input features.
    n_features = len(feature_columns) + 1

    print(f"   Features: {len(feature_columns)} indicators + log_return (target)")
    print(f"   Note: Adj Close replaced by Log_Ret_Lag1 for stationarity and cross-model consistency")
    print(f"   Total observations after feature engineering: {len(df)}")

    print(f"\n[SCALING] Scaling features...")

    if len(df) < 10:
        print(f"   ✗ Insufficient data for {ticker}"); continue

    split_idx  = int(len(df) * TRAIN_RATIO)
    train_size = split_idx
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[:train_size] = True
    test_mask = np.logical_not(train_mask)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df.loc[train_mask, feature_columns + ['log_return']].values)
    scaled_data  = scaler.transform(df[feature_columns + ['log_return']].values)
    ticker_clean = ticker.replace(".", "_")
    os.makedirs('saved_models', exist_ok=True)
    joblib_dump(scaler, f'saved_models/scaler_{ticker_clean}_gru.pkl')
    print(f'   - Saved scaler: scaler_{ticker_clean}_gru.pkl')

    train_data = scaled_data[train_mask]
    test_data  = scaled_data[test_mask]

    print(f"   - Features normalised to [0, 1] using training-set statistics only")
    print(f"   Train : {len(train_data)} obs  ({df.index[train_mask][0].date()} → {df.index[train_mask][-1].date()})")
    print(f"   Test  : {len(test_data)} obs  ({df.index[test_mask][0].date()} → {df.index[test_mask][-1].date()})")
    print(f"   Ratio : {len(train_data)/(len(train_data)+len(test_data)):.4f} / "
          f"{len(test_data)/(len(train_data)+len(test_data)):.4f}")

    train_prices_all = df.loc[train_mask, 'Adj Close'].values
    test_prices_all  = df.loc[test_mask,  'Adj Close'].values

    # ══════════════════════════════════════════════════════════════════════
    #   GRID SEARCH — HYPERPARAMETER OPTIMISATION
    #   Tries up to 50 random combinations from the param_grid search space.
    #   get_units_for_lookback constrains which unit values are tried for
    #   each lookback length — this is documented in the dissertation as a
    #   deliberate choice to keep the search manageable while avoiding
    #   networks that are clearly too small or too large for the sequence
    #   length. Early stopping uses val_loss (not training loss) to prevent
    #   the model from overfitting during the search.
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[GRID SEARCH] Running randomised search for GRU hyperparameters...")

    best_val_rmse = float('inf')
    best_config   = None
    best_model    = None
    best_history  = None
    all_configs   = []

    all_combos = []
    for lb in param_grid['lookback']:
        for u in get_units_for_lookback(lb):
            for l in param_grid['layers']:
                for dr in param_grid['dropout']:
                    for lr_rate in param_grid['lr']:
                        for bs in param_grid['batch_size']:
                            all_combos.append((lb, u, l, dr, lr_rate, bs))

    random.seed(42)
    random.shuffle(all_combos)
    max_trials = min(50, len(all_combos))
    print(f"   Testing {max_trials} random combinations (from {len(all_combos)} total)...")
    print(f"   Note: unit search is constrained per lookback to prevent under/over-parameterisation.")

    for combo_count, (lb, u, l, dr, lr_rate, bs) in enumerate(all_combos[:max_trials], start=1):
        try:
            X_train_seq, y_train_arr = create_sequences(train_data, lb)
            if len(X_train_seq) == 0:
                print(f"    Skipping LB={lb} → not enough training data")
                continue

            candidate_model = build_gru_model(lb, n_features, u, l, dr, lr_rate)
            training_hist   = candidate_model.fit(
                X_train_seq, y_train_arr,
                epochs=param_grid['epochs'][0],
                batch_size=bs,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=8,
                                         restore_best_weights=True)],
                verbose=0)

            val_size         = max(1, int(len(X_train_seq) * 0.2))
            val_preds_scaled = candidate_model.predict(X_train_seq[-val_size:], verbose=0).flatten()
            val_true_scaled  = y_train_arr[-val_size:]
            val_rmse         = np.sqrt(mean_squared_error(val_true_scaled, val_preds_scaled))

            config_str = f"LB:{lb}|U:{u}|L:{l}|DR:{dr}|LR:{lr_rate}|BS:{bs}"
            all_configs.append({
                'Ticker': ticker, 'lookback': lb, 'units': u,
                'layers': l, 'dropout': dr, 'lr': lr_rate,
                'batch_size': bs, 'val_rmse_logret': val_rmse})

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_config   = {'lookback': lb, 'units': u, 'layers': l,
                                 'dropout': dr, 'lr': lr_rate, 'batch_size': bs,
                                 'config_str': config_str}
                best_model    = candidate_model
                best_history  = training_hist
            else:
                del candidate_model
                keras.backend.clear_session()

        except Exception as e:
            print(f"    Error with LB={lb}, U={u}: {e}")
            continue

        if combo_count % 5 == 0:
            print(f"   Progress: {combo_count}/{max_trials} combinations...")

    print(f"   - Randomised search complete: {len(all_configs)} valid configurations")

    grid_df = pd.DataFrame(all_configs).sort_values('val_rmse_logret')
    grid_df.to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_grid_search_gru.csv"), index=False)
    print(f"   - Saved: {ticker}_grid_search_gru.csv")

    display_cols = ['lookback', 'units', 'layers', 'dropout', 'lr', 'val_rmse_logret']
    print(f"\n   Top 5 Configurations:")
    print(tabulate(grid_df.head(5)[display_cols], headers='keys',
                   tablefmt='simple', showindex=False, floatfmt='.6f'))

    pd.DataFrame([{'Ticker': ticker, 'Model': 'GRU',
                   'lookback': best_config['lookback'], 'units': best_config['units'],
                   'layers': best_config['layers'],     'dropout': best_config['dropout'],
                   'lr': best_config['lr'],             'batch_size': best_config['batch_size'],
                   'Val_RMSE_LogRet': best_val_rmse}]) \
      .to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_best_hyperparameters.csv"), index=False)
    print(f"   - Saved: {ticker}_best_hyperparameters.csv")
    print(f"\n   Best Configuration: {best_config['config_str']}")
    print(f"   Validation RMSE (log-return, scaled): {best_val_rmse:.6f}")

    # ══════════════════════════════════════════════════════════════════════
    #   UNCERTAINTY SETUP
    #   Works out the residual standard deviation from the validation slice
    #   of the training data. This is then used to build the 95% prediction
    #   intervals during walk-forward and for the 5 day forecast. The std
    #   is also converted from scaled space back to real log-return space
    #   so the forecast intervals are in meaningful price units.
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[UNCERTAINTY SETUP] Computing residual std from validation set...")

    lb_best                            = best_config['lookback']
    X_train_seq_full, y_train_arr_full = create_sequences(train_data, lb_best)
    val_size_for_std                   = max(1, int(len(X_train_seq_full) * 0.2))
    val_X_for_std                      = X_train_seq_full[-val_size_for_std:]
    val_y_for_std                      = y_train_arr_full[-val_size_for_std:]
    val_preds_for_std                  = best_model.predict(val_X_for_std, verbose=0).flatten()
    val_residuals                      = val_y_for_std - val_preds_for_std
    residual_std                       = max(np.std(val_residuals, ddof=1), 1e-6)
    print(f"   Residual std from validation set (scaled space): {residual_std:.6f}")

    dummy_std_pos        = np.zeros((1, scaled_data.shape[1]))
    dummy_std_neg        = np.zeros((1, scaled_data.shape[1]))
    dummy_std_pos[0, -1] = residual_std
    dummy_std_neg[0, -1] = 0.0
    real_residual_std    = abs(scaler.inverse_transform(dummy_std_pos)[0, -1] -
                               scaler.inverse_transform(dummy_std_neg)[0, -1])
    print(f"   Residual std in real log-return space:            {real_residual_std:.6f}")

    if not (0.0001 < real_residual_std < 0.5):
        print(f"   WARNING: real_residual_std={real_residual_std:.6f} is outside the expected"
              f" range [0.0001, 0.5] for daily log-returns. Check scaler inversion.")

    # ══════════════════════════════════════════════════════════════════════
    #   WALK-FORWARD VALIDATION
    #   Steps through the test set one day at a time, making a next-day
    #   prediction at each step and then adding the true value to the
    #   history before moving on. The model gets refitted every 63 trading
    #   days to match the ARIMA and RF refit schedules. Early stopping
    #   during refits uses val_loss rather than training loss to stop the
    #   model overfitting to the most recent data.
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[WALK-FORWARD] Running walk-forward validation "
          f"(refit every {WALK_REFIT_FREQ} days)...")

    lb          = best_config['lookback']
    history_seq = list(train_data)
    wf_mean, wf_lower, wf_upper = [], [], []

    def inv_log(arr):
        dummy = np.zeros((len(arr), scaled_data.shape[1]))
        dummy[:, -1] = np.array(arr)
        return scaler.inverse_transform(dummy)[:, -1]

    for step_idx in range(len(test_data)):
        current_history   = np.array(history_seq)
        last_sequence     = current_history[-lb:].reshape(1, lb, n_features)

        mean_scaled, lower_scaled, upper_scaled = get_prediction_intervals(
            best_model, last_sequence, residual_std, horizon=1)

        wf_mean.append(inv_log(mean_scaled)[0])
        wf_lower.append(inv_log(lower_scaled)[0])
        wf_upper.append(inv_log(upper_scaled)[0])

        history_seq.append(test_data[step_idx])

        if step_idx > 0 and step_idx % WALK_REFIT_FREQ == 0:
            X_refit, y_refit = create_sequences(np.array(history_seq), lb)
            best_model.fit(
                X_refit, y_refit,
                epochs=10,
                batch_size=best_config['batch_size'],
                validation_split=0.1,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3,
                                         restore_best_weights=True)],
                verbose=0)

        if (step_idx + 1) % 50 == 0:
            print(f"   Progress: {step_idx+1}/{len(test_data)} predictions...")

    wf_predicted_logrets    = np.array(wf_mean)
    wf_ci_lower_logrets     = np.array(wf_lower)
    wf_ci_upper_logrets     = np.array(wf_upper)

    last_train_price = train_prices_all[-1]
    base_prices      = np.concatenate([[last_train_price], test_prices_all[:-1]])

    wf_predicted_prices = base_prices * np.exp(wf_predicted_logrets)
    wf_ci_lower_prices  = base_prices * np.exp(wf_ci_lower_logrets)
    wf_ci_upper_prices  = base_prices * np.exp(wf_ci_upper_logrets)

    test_actual = test_prices_all
    test_pred   = wf_predicted_prices
    test_dates  = df.index[test_mask]
    print(f"   - Walk-forward complete: {len(test_pred)} next-day predictions with 95% PIs")

    # ══════════════════════════════════════════════════════════════════════
    #   PERFORMANCE METRICS
    #   Standard out-of-sample metrics calculated against the walk-forward
    #   predictions. OOS RMSE on log-returns is the scale-free metric used
    #   for cross-ticker and cross-model comparison in the dissertation.
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[METRICS] Calculating performance metrics...")

    mape_oos = mean_absolute_percentage_error(test_actual, test_pred)
    rmse     = np.sqrt(mean_squared_error(test_actual, test_pred))
    mae      = mean_absolute_error(test_actual, test_pred)

    test_log_rets   = df.loc[test_mask, 'log_return'].values
    oos_rmse_logret = np.sqrt(mean_squared_error(test_log_rets, wf_predicted_logrets))

    actual_direction    = np.sign(np.diff(test_actual))
    predicted_direction = np.sign(np.diff(test_pred))
    dir_accuracy        = np.mean(actual_direction == predicted_direction) * 100

    # ══════════════════════════════════════════════════════════════════════
    #   UNCERTAINTY METRICS — RAW AND NORMALISED WINKLER SCORE
    # ══════════════════════════════════════════════════════════════════════

    interval_widths    = wf_ci_upper_prices - wf_ci_lower_prices
    avg_interval_width = np.mean(interval_widths)
    coverage_rate      = np.mean((test_actual >= wf_ci_lower_prices) &
                                 (test_actual <= wf_ci_upper_prices)) * 100
    winkler_raw        = winkler_score(test_actual, wf_ci_lower_prices, wf_ci_upper_prices, alpha=0.05)
    winkler_norm       = winkler_score_normalised(test_actual, wf_ci_lower_prices, wf_ci_upper_prices, alpha=0.05)
    mean_actual_price  = np.mean(test_actual)

    print(f"\n   Performance Metrics (Out-of-Sample, Next-Day Walk-Forward):")
    print(f"   {'─'*50}")
    print(f"   MAPE:                    {mape_oos:.4%}")
    print(f"   RMSE:                    {rmse:.4f}")
    print(f"   MAE:                     {mae:.4f}")
    print(f"   OOS RMSE (log-return):   {oos_rmse_logret:.6f}  ← scale-free, comparable across tickers")
    print(f"   Directional Accuracy:    {dir_accuracy:.2f}%")
    print(f"\n   Uncertainty Metrics (95% PI — empirical residual-std-based):")
    print(f"   NOTE: ARIMA uses analytical CIs; comparison is indicative only.")
    print(f"   {'─'*50}")
    print(f"   Avg Interval Width:         {avg_interval_width:.4f} GBX")
    print(f"   Coverage Rate:              {coverage_rate:.2f}%")
    print(f"   Mean Actual Price (test):   {mean_actual_price:.4f} GBX")
    print(f"   Winkler Score (95%, raw):   {winkler_raw:.4f}  ← GBX units, within-ticker comparison only")
    print(f"   Winkler Score (95%, norm):  {winkler_norm:.6f}  ← price-scaled, valid cross-ticker comparison")

    all_uncertainty_metrics.append({
        'Ticker': ticker, 'Model': 'GRU',
        'Avg_Interval_Width': avg_interval_width,
        'Coverage_Rate_%': coverage_rate,
        'Mean_Actual_Price': mean_actual_price,
        'Winkler_Score_Raw': winkler_raw,
        'Winkler_Score_Norm': winkler_norm,
        'Dir_Accuracy_%': dir_accuracy,
        'PI_Method': 'Empirical (residual std)'})

    if len(test_actual) > 1:
        gru_errors      = test_actual[1:] - test_pred[1:]
        naive_errors    = test_actual[1:] - test_actual[:-1]
        dm_stat, dm_pval       = diebold_mariano_test(gru_errors, naive_errors)
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
        pd.DataFrame([{'Ticker': ticker, 'Model': f'GRU_{best_config["config_str"]}',
                       'DM_Statistic': dm_stat, 'DM_pValue': dm_pval,
                       'Significant_005': dm_significance}]) \
          .to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_diebold_mariano.csv"), index=False)
        print(f"   - Saved: {ticker}_diebold_mariano.csv")

    # ══════════════════════════════════════════════════════════════════════
    #   FEATURE IMPORTANCE
    #   Uses permutation importance — one feature column at a time gets
    #   shuffled across all test sequences and the resulting drop in MAPE
    #   is recorded. A bigger drop means the model relied on that feature
    #   more heavily. The shuffling breaks cross-sample relationships while
    #   keeping the order within each window intact. This limitation is
    #   acknowledged in the dissertation methodology chapter.
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[FEATURE IMPORTANCE] Calculating permutation-based feature importance...")
    extended_test   = np.concatenate([train_data[-lb:], test_data])
    X_test_full     = np.array([extended_test[i-lb:i] for i in range(lb, len(extended_test))])
    baseline_mape   = mape_oos
    fi_scores       = []

    for feat_idx, feature_name in enumerate(feature_columns):
        X_perm = X_test_full.copy()
        np.random.shuffle(X_perm[:, :, feat_idx])
        y_perm_scaled = best_model.predict(X_perm, verbose=0)
        dummy         = np.zeros((len(y_perm_scaled), scaled_data.shape[1]))
        dummy[:, -1]  = y_perm_scaled.flatten()
        y_perm_log    = scaler.inverse_transform(dummy)[:, -1]
        y_perm_prices = base_prices * np.exp(y_perm_log)
        importance    = mean_absolute_percentage_error(test_prices_all, y_perm_prices) - baseline_mape
        fi_scores.append(importance)
        print(f"   {feature_name:15} → Importance: {importance:.6f}")

    pd.DataFrame({'Feature': feature_columns, 'Importance': fi_scores}) \
      .sort_values('Importance', ascending=False) \
      .assign(Ticker=ticker, Model='GRU') \
      .to_csv(os.path.join(PER_TICKER_FOLDER, f"{ticker}_feature_importance.csv"), index=False)
    print(f"   - Saved: {ticker}_feature_importance.csv")

    # ══════════════════════════════════════════════════════════════════════
    #   DIAGNOSTIC PLOTS & 5-DAY FORECAST
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n[PLOTS & FORECAST] Generating diagnostic plots and 5-day forecast...")
    create_prediction_plot_with_ci(test_actual, test_pred,
                                   wf_ci_lower_prices, wf_ci_upper_prices,
                                   ticker, PLOTS_FOLDER, test_dates)
    create_residual_acf_plot(test_actual - test_pred, ticker, PLOTS_FOLDER)
    create_feature_importance_plot(np.array(fi_scores), feature_columns, ticker, PLOTS_FOLDER)
    create_training_history_plot(best_history, ticker, best_config['config_str'], PLOTS_FOLDER)

    last_price     = df['Adj Close'].iloc[-1]
    last_date      = df.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)
    day_weights    = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

    # ══════════════════════════════════════════════════════════════════════
    #   5-DAY FORECAST
    #   Recursive forecast — each day's predicted log-return gets converted
    #   back to real space, used to advance the price, then re-scaled before
    #   being fed into the next lookback window. real_residual_std is used
    #   for the interval widths rather than the scaled version, so the bands
    #   are in real log-return space and widen correctly with the horizon.
    # ══════════════════════════════════════════════════════════════════════

    last_X = scaled_data[-lb_best:].copy().reshape(1, lb_best, n_features)

    forecast_log_rets = []

    for h in range(5):
        mean_scaled, _, _ = get_prediction_intervals(
            best_model, last_X, residual_std, horizon=1)

        # Convert from scaled [0,1] space back to real log-return space
        dummy_inv        = np.zeros((1, scaled_data.shape[1]))
        dummy_inv[0, -1] = mean_scaled[0]
        real_log_ret     = scaler.inverse_transform(dummy_inv)[0, -1]
        forecast_log_rets.append(real_log_ret)

        # Re-scale before feeding into the next lookback window
        dummy_fwd        = np.zeros((1, scaled_data.shape[1]))
        dummy_fwd[0, -1] = real_log_ret
        scaled_log_ret   = scaler.transform(dummy_fwd)[0, -1]

        # Roll the lookback window forward
        next_row      = last_X[0, -1, :].copy()
        next_row[9]   = scaled_log_ret   # Log_Ret_Lag1 (feature index 9)
        next_row[-1]  = scaled_log_ret   # log_return   (target column, last index)
        last_X        = np.concatenate(
            [last_X[:, 1:, :], next_row.reshape(1, 1, -1)], axis=1)

    forecast_log_rets   = np.array(forecast_log_rets)
    cumulative_log_rets = np.cumsum(forecast_log_rets)
    horizons            = np.arange(1, 6)
    forecast_prices     = last_price * np.exp(cumulative_log_rets)
    forecast_ci_lower   = last_price * np.exp(cumulative_log_rets - 1.96 * real_residual_std * np.sqrt(horizons))
    forecast_ci_upper   = last_price * np.exp(cumulative_log_rets + 1.96 * real_residual_std * np.sqrt(horizons))

    # The weighted average gives more importance to the nearer days since
    # short-term forecasts from any model tend to be more reliable than
    # longer ones. Day 1 gets 50% of the weight, day 2 gets 20%, and the
    # remaining three days split the last 30% equally between them.
    weighted_avg    = np.sum(forecast_prices * day_weights)
    weighted_signal = "UP" if weighted_avg > last_price else "DOWN"

    print(f"\n   Next 5 Trading Days Forecast:")
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
        print(f"      95% PI:    [{ci_low:.3f}, {ci_up:.3f}] GBP (residual-based, width ∝ √horizon)")
        ticker_forecast_rows.append({
            'Ticker': ticker, 'Day': i+1, 'Date': fdate.date(),
            'Forecast_Price': fprice, 'Weight': weight,
            'Weighted_Contribution': fprice * weight,
            'PI_Lower': ci_low, 'PI_Upper': ci_up, 'Signal': signal})

    pd.DataFrame(ticker_forecast_rows).to_csv(
        os.path.join(PER_TICKER_FOLDER, f"{ticker}_5day_forecast.csv"), index=False)
    print(f"\n   - Saved: {ticker}_5day_forecast.csv")

    create_forecast_chart(
        ticker, df, 'Adj Close', last_price, last_date,
        forecast_dates, forecast_prices, forecast_ci_lower, forecast_ci_upper,
        model_label='GRU', output_folder=PLOTS_FOLDER, n_history=30)

    all_forecasts.append([
        ticker, f"{last_price:.3f}",
        f"{forecast_prices[0]:.3f}", f"{forecast_prices[-1]:.3f}",
        f"{weighted_avg:.3f}", weighted_signal,
        f"[{forecast_ci_lower[0]:.3f}, {forecast_ci_upper[0]:.3f}]",
        f"[{forecast_ci_lower[-1]:.3f}, {forecast_ci_upper[-1]:.3f}]"])

    config_str = (f"lb={best_config['lookback']}|units={best_config['units']}|"
                  f"layers={best_config['layers']}|dropout={best_config['dropout']}|"
                  f"lr={best_config['lr']}")
    all_performance_rows.append({
        'Ticker':           ticker,
        'Model':            'GRU',
        'Val_RMSE_LogRet':  f"{best_val_rmse:.6f}",
        'OOS_RMSE_LogRet':  f"{oos_rmse_logret:.6f}",
        'OOS_MAPE':         f"{mape_oos:.4%}",
        'RMSE':             f"{rmse:.4f}",
        'MAE':              f"{mae:.4f}",
        'Dir_Accuracy':     f"{dir_accuracy:.2f}%",
        'Best_Config':      config_str,
        'Refit_Freq':       f"Every {WALK_REFIT_FREQ} days",
        'Test_Start':       test_dates[0].date(),
        'Test_End':         test_dates[-1].date(),
        'Test_Obs':         len(test_actual)})

    os.makedirs('saved_models', exist_ok=True)
    best_model.save(f'saved_models/gru_{ticker.replace(".", "_")}.keras')
    print(f"   - Saved model: gru_{ticker.replace('.', '_')}.keras")

    keras.backend.clear_session()

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

print("\n" + "═" * 100)
print("GRU MODEL - PERFORMANCE SUMMARY".center(100))
print("═" * 100 + "\n")

performance_df = pd.DataFrame(all_performance_rows)
print(tabulate(performance_df, headers='keys', tablefmt='grid', showindex=False))
performance_df.to_csv(os.path.join(SUMMARY_FOLDER, "GRU_performance_summary.csv"), index=False)
print(f"\n- Saved: GRU_performance_summary.csv")

uncertainty_df = pd.DataFrame(all_uncertainty_metrics)
print("\nUncertainty Quantification Summary:")
print(tabulate(uncertainty_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))
uncertainty_df.to_csv(os.path.join(SUMMARY_FOLDER, "GRU_uncertainty_metrics.csv"), index=False)
print(f"\n- Saved: GRU_uncertainty_metrics.csv")

print("\n5-Day Price Forecasts with 95% PI (residual-based, width ∝ √horizon):")
print(tabulate(all_forecasts,
               headers=['Ticker', 'Last Actual', 'Day 1', 'Day 5',
                        'Weighted Avg', 'Trend', 'Day 1 PI', 'Day 5 PI'],
               tablefmt='grid'))
forecast_df = pd.DataFrame(all_forecasts,
    columns=['Ticker', 'Last_Actual_Price', 'Day1_Forecast', 'Day5_Forecast',
             'Weighted_Avg', 'Trend', 'Day1_PI', 'Day5_PI'])
forecast_df.insert(1, 'Model', 'GRU')
forecast_df.to_csv(os.path.join(SUMMARY_FOLDER, "GRU_5day_forecasts_summary.csv"), index=False)
print(f"\n- Saved: GRU_5day_forecasts_summary.csv")

sys.stdout = _real_stdout
log_file.close()
