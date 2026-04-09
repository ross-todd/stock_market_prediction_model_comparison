# Stock Market Prediction — Comparative Analysis

BSc (Hons) Software Development — Honours Project 2026
By Ross Todd

A comparative analysis of three forecasting models — ARIMA, Random Forest, and GRU — applied to three UK banking stocks: Barclays (BARC.L), Lloyds (LLOY.L), and HSBC (HSBA.L). Each model is evaluated using walk-forward validation over a fixed 2021–2026 data window with a shared set of performance and uncertainty metrics.

---

## Models

| Model         | Type                  | Features                            |
| ------------- | --------------------- | ----------------------------------- |
| ARIMA         | Classical statistical | Univariate — log-returns only      |
| Random Forest | Machine learning      | 14 engineered technical indicators  |
| GRU           | Deep learning         | 10 engineered features + log-return |

ARIMA acts as the baseline. Random Forest and GRU are both benchmarked against it. All three use the same data window, 80/20 train/test split, walk-forward protocol, and refit frequency (every 63 trading days) so that any difference in results is down to the model itself.

---

## Project Structure

```
├── data_loader.py          # Downloads and cleans OHLCV data from Yahoo Finance
├── arima_model.py          # ARIMA walk-forward validation and 5-day forecast
├── rf_model.py             # Random Forest walk-forward validation and 5-day forecast
├── gru_model.py            # GRU walk-forward validation and 5-day forecast
├── arima_results/          # ARIMA output — plots, per-ticker CSVs, summary tables
├── rf_results/             # RF output — plots, per-ticker CSVs, summary tables
├── gru_results/            # GRU output — plots, per-ticker CSVs, summary tables
├── saved_models/           # Saved RF (.pkl) and GRU (.keras) models and scalers
└── requirements.txt        # Python dependencies
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/ross-todd/stock_market_prediction_model_comparison.git
cd stock_market_prediction_model_comparison
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Models

Each model script runs independently. Run them in order if you want the saved models from RF and GRU to be available for the Streamlit application.

```bash
python arima_model.py
python rf_model.py
python gru_model.py
```

Each script will:

- Download data automatically via `data_loader.py`
- Run a hyperparameter search to find the best configuration
- Run walk-forward validation across the test set
- Save all results, plots, and summary CSVs to the relevant results folder
- Print everything to the terminal and save it to a `terminal_output.txt` log file

---

## Output

Each model produces the same set of outputs so results are directly comparable.

**Per ticker:**

- Actual vs predicted chart with 95% prediction interval
- Training history plot (GRU only)
- 30-day history + 5-day forecast chart
- Grid search results CSV
- Best hyperparameters CSV
- 5-day forecast CSV
- Diebold-Mariano test CSV

**Summary (all three tickers combined):**

- `*_performance_summary.csv` — MAPE, RMSE, MAE, directional accuracy
- `*_uncertainty_metrics.csv` — coverage rate, interval width, Winkler score
- `*_5day_forecasts_summary.csv` — 5-day forecasts with weighted average signals

**Combined Charts**

* Coverage
* Forecasts
* Out-Of-Sample MAPE Comparison
* Prediction Uncertainty

---

## Requirements

- Python 3.9 or higher
- See `requirements.txt` for full list of dependencies

## Pre-trained Models (Optional)

The trained Random Forest and GRU models are not included in this repository due to file size constraints.

They can be downloaded here:

* [saved_models](https://caledonianac-my.sharepoint.com/:f:/g/personal/rtodd303_caledonian_ac_uk/IgAs84BaZv07RKhJjEWjjADJAS3Rb4HTriC91nE08D9eP6I?e=IDXicD)

Once downloaded, place them in the `saved_models/` directory before running the random_forest_analysis.py, and teh gru_analysis.py files

---

## Evaluation Methodology

All three models are evaluated using the same metrics:

| Metric                     | Description                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| MAPE                       | Mean Absolute Percentage Error — average % difference between predicted and actual prices |
| RMSE                       | Root Mean Squared Error — penalises large errors more heavily                             |
| MAE                        | Mean Absolute Error — average absolute difference in pence                                |
| OOS RMSE (log-return)      | Scale-free RMSE on log-returns, comparable across all three stocks                         |
| Directional Accuracy       | % of days the model correctly predicted whether the price went up or down                  |
| Coverage Rate              | % of actual prices that fell inside the 95% prediction interval                            |
| Winkler Score (normalised) | Interval quality score divided by mean price — allows fair cross-stock comparison         |
| Diebold-Mariano Test       | Checks whether model performance is significantly better than a naive random walk          |

---

## Disclaimer

This code is for educational and research purposes only. Do not use it as the basis for any real investment decisions.
