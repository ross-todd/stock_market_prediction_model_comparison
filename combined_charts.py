# Ross Todd
# BSc (Hons) Software Development
# Honours Project 2026 - Stock Market Prediction Comparison Analysis
#
# combined_charts.py - Chart and Analysis Generator
#
# --- Outputs ---
# 1. Forecast PNGs: 20-day history + 5-day pred vs actual March prices (per model)
# 2. MAPE Comparison: Bar chart showing OOS performance across all stocks
# 3. Stats Audit: Console output of MAE/MAPE and daily error decay
# 4. Confidence Intervals: 5-day forecast with CI shading and price labels
# 5. Coverage Maps: Full test period showing actual vs pred + highlighted CI breakouts

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

BASE_DIR     = "."
SAVED_DIR    = "saved_data"
OUTPUT_DIR   = "combined_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "ARIMA":         "arima_results",
    "Random Forest": "rf_results",
    "GRU":           "gru_results",
}

# Prediction CSV filename suffixes per model
PRED_SUFFIXES = {
    "ARIMA":         "ARIMA_predictions.csv",
    "Random Forest": "RF_predictions.csv",
    "GRU":           "GRU_predictions.csv",
}

STOCKS = {
    "Barclays": {
        "forecast_ticker": "BARC.L",
        "saved_file":      "BARC_L_20210228_20260228.csv",
    },
    "Lloyds": {
        "forecast_ticker": "LLOY.L",
        "saved_file":      "LLOY_L_20210228_20260228.csv",
    },
    "HSBC": {
        "forecast_ticker": "HSBA.L",
        "saved_file":      "HSBA_L_20210228_20260228.csv",
    },
}

SUBPLOT_TITLES = [
    "Barclays (BARC.L)",
    "Lloyds (LLOY.L)",
    "HSBC (HSBA.L)",
]

STOCK_COLOURS = {
    "Barclays": "#1f77b4",
    "Lloyds":   "#2ca02c",
    "HSBC":     "#d62728",
}

FORECAST_COLOUR = "orange"
FORECAST_FILL   = "rgba(255,165,0,0.18)"  # lighter shading for 5-day forecast charts
COVERAGE_FILL   = "rgba(255,165,0,0.35)"  # darker shading for coverage charts
OUTSIDE_COLOUR  = "rgba(180,0,0,1.0)"     # fully opaque red for points outside CI

DATE_COL     = "Date"
FORECAST_COL = "Forecast_Price"

# Actual March 2026 prices
ACTUAL_MARCH = {
    "Barclays": [437.35, 422.55, 430.60, 417.00, 404.15],
    "Lloyds":   [99.92,  96.94,  98.28,  96.90,  95.42],
    "HSBC":     [1332.00, 1262.80, 1291.40, 1278.80, 1245.00],
}

MARCH_DATES = ["02 Mar", "03 Mar", "04 Mar", "05 Mar", "06 Mar"]

# Last actual closing prices on 27 Feb 2026
BASE_PRICES = {
    "Barclays": 452.85,
    "Lloyds":   102.45,
    "HSBC":     1393.60,
}

# Hardcoded forecasts for MAPE analysis
FORECASTS_MAPE = {
    "ARIMA": {
        "BARC.L": [453.40, 454.12, 454.85, 455.58, 456.31],
        "LLOY.L": [102.38, 102.54, 102.66, 102.80, 102.94],
        "HSBA.L": [1395.14, 1395.86, 1397.12, 1398.03, 1399.17],
    },
    "RF": {
        "BARC.L": [453.08, 453.32, 453.55, 453.79, 454.02],
        "LLOY.L": [102.51, 102.56, 102.61, 102.67, 102.72],
        "HSBA.L": [1394.67, 1395.74, 1396.81, 1397.89, 1398.96],
    },
    "GRU": {
        "BARC.L": [458.14, 462.88, 467.04, 470.78, 474.26],
        "LLOY.L": [103.10, 103.58, 104.01, 104.39, 104.74],
        "HSBA.L": [1398.33, 1402.55, 1406.69, 1410.98, 1415.41],
    },
}

ACTUAL_MAPE = {
    "BARC.L": [437.35, 422.55, 430.60, 417.00, 404.15],
    "LLOY.L": [99.92,  96.94,  98.28,  96.90,  95.42],
    "HSBA.L": [1332.00, 1262.80, 1291.40, 1278.80, 1245.00],
}

# --- HELPER FUNCTIONS ---

def load_actual(saved_file, n_days=20):
    path = os.path.join(BASE_DIR, SAVED_DIR, saved_file)
    df   = pd.read_csv(path, parse_dates=[DATE_COL])
    df   = df.sort_values(DATE_COL).tail(n_days).reset_index(drop=True)
    return df

def load_forecast(model_folder, forecast_ticker):
    path = os.path.join(
        BASE_DIR, model_folder, "per_ticker_results",
        f"{forecast_ticker}_5day_forecast.csv"
    )
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df

def load_predictions(model_folder, forecast_ticker, model_suffix):
    path = os.path.join(
        BASE_DIR, model_folder, "per_ticker_results",
        f"{forecast_ticker}_{model_suffix}"
    )
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df

def get_lower_col(df):
    for col in ["CI_Lower", "PI_Lower", "ci_lower", "pi_lower"]:
        if col in df.columns:
            return col
    return None

def get_upper_col(df):
    for col in ["CI_Upper", "PI_Upper", "ci_upper", "pi_upper"]:
        if col in df.columns:
            return col
    return None

def weekly_ticks(dates):
    return [dates[i] for i in range(0, len(dates), 5)]


# --- SECTION 1: FORECAST CHARTS ---
# One figure per model, three subplots (one per stock)
# Shows last 20 trading days of actual price, 5-day forecast line,
# and actual March 2026 prices for comparison

print("\n" + "="*60)
print("SECTION 1 - Building forecast charts")
print("="*60)

for model_name, model_folder in MODELS.items():
    print(f"  Building figure for {model_name}...")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=SUBPLOT_TITLES,
        vertical_spacing=0.18,
    )

    for row, (stock_name, stock_info) in enumerate(STOCKS.items(), start=1):
        actual_colour   = STOCK_COLOURS[stock_name]
        forecast_ticker = stock_info["forecast_ticker"]
        saved_file      = stock_info["saved_file"]

        # Load actual prices for last 20 trading days
        try:
            actual_df     = load_actual(saved_file)
            actual_dates  = actual_df[DATE_COL].dt.strftime("%d %b").tolist()
            actual_prices = actual_df["Close"].tolist()
            last_date     = actual_dates[-1]
            last_price    = actual_prices[-1]
        except Exception as e:
            print(f"    WARNING: could not load actual data for {stock_name}: {e}")
            actual_dates, actual_prices, last_date, last_price = [], [], None, None

        # Load 5-day forecast
        fc_dates, fc_prices = [], []
        try:
            fc        = load_forecast(model_folder, forecast_ticker)
            fc_dates  = fc[DATE_COL].dt.strftime("%d %b").tolist()
            fc_prices = fc[FORECAST_COL].tolist()
        except FileNotFoundError:
            print(f"    WARNING: forecast not found for {model_name} / {forecast_ticker}")

        march_prices = ACTUAL_MARCH[stock_name]

        # Plot actual historical line
        if actual_dates:
            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=actual_prices,
                mode="lines",
                name=stock_name,
                line=dict(color=actual_colour, width=3),
                showlegend=True,
                legendgroup=stock_name,
                legendrank=row,
                hovertemplate="%{y:.2f}p<extra>Actual</extra>",
            ), row=row, col=1)

        # Plot forecast line, connecting from last actual price
        if fc_dates and last_date:
            fig.add_trace(go.Scatter(
                x=[last_date] + fc_dates,
                y=[last_price] + fc_prices,
                mode="lines+markers",
                name="Forecast",
                line=dict(color=FORECAST_COLOUR, width=3.5, dash="dash"),
                marker=dict(size=6, color=FORECAST_COLOUR),
                showlegend=(row == 1 and stock_name == "Barclays"),
                legendgroup="forecast",
                legendrank=1000,
                hovertemplate="%{y:.2f}p<extra>Forecast</extra>",
            ), row=row, col=1)

        # Plot actual March prices for comparison
        if last_date and march_prices:
            fig.add_trace(go.Scatter(
                x=[last_date] + MARCH_DATES,
                y=[last_price] + march_prices,
                mode="lines+markers",
                name="Actual (Mar)",
                line=dict(color=FORECAST_COLOUR, width=3),
                marker=dict(size=6, color=FORECAST_COLOUR),
                showlegend=(row == 1 and stock_name == "Barclays"),
                legendgroup="actual_march",
                legendrank=999,
                hovertemplate="%{y:.2f}p<extra>Actual Mar</extra>",
            ), row=row, col=1)

        # Vertical line marking start of forecast period
        if last_date:
            fig.add_vline(
                x=last_date,
                line_width=1,
                line_dash="dot",
                line_color="grey",
                row=row, col=1
            )

        # Set weekly x-axis ticks
        if actual_dates:
            all_dates = actual_dates + (fc_dates if fc_dates else [])
            tick_vals = weekly_ticks(all_dates)
            fig.update_xaxes(
                tickvals=tick_vals,
                ticktext=tick_vals,
                tickangle=-45,
                row=row, col=1
            )

        fig.update_yaxes(
            title_text="Price (GBX)",
            showgrid=True,
            gridcolor="#eeeeee",
            row=row, col=1
        )

    fig.update_layout(
        title=dict(
            text=f"{model_name} — Last 20 Trading Days, 5-Day Forecast & Actual",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
            y=0.98,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1188,
        width=840,
        margin=dict(l=70, r=40, t=130, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")

    safe_name = model_name.lower().replace(" ", "_")
    out_path  = os.path.join(OUTPUT_DIR, f"forecast_{safe_name}.png")
    fig.write_image(out_path, scale=2)
    print(f"    Saved: {out_path}")


# --- SECTION 2: OOS MAPE BAR CHART ---
# Grouped bar chart showing OOS MAPE by model and stock (Table 4.5 values)

print("\n" + "="*60)
print("SECTION 2 - Building OOS MAPE bar chart")
print("="*60)

models_list  = ["ARIMA", "Random Forest", "GRU"]
tickers_list = ["Barclays", "Lloyds", "HSBC"]
bar_colors   = ["#1f77b4", "#2ca02c", "#d62728"]

mape_values = {
    "Barclays": [1.4185, 1.3745, 1.3996],
    "Lloyds":   [1.1529, 1.1150, 1.2524],
    "HSBC":     [1.1228, 1.0344, 1.0496],
}

x     = np.arange(len(models_list))
width = 0.25

fig2, ax = plt.subplots(figsize=(10, 6))

for i, (ticker, color) in enumerate(zip(tickers_list, bar_colors)):
    bars = ax.bar(x + i * width, mape_values[ticker], width,
                  label=ticker, color=color, edgecolor="white", linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=9)

ax.set_xlabel("\nModel", fontsize=11)
ax.set_ylabel("OOS MAPE (%)", fontsize=11)
ax.set_title("Out-of-Sample MAPE by Model and Stock", fontsize=13, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(models_list)
ax.legend(loc="upper right")
ax.set_ylim(0, 1.7)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
mape_path = os.path.join(OUTPUT_DIR, "OOS_MAPE_comparison.png")
plt.savefig(mape_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {mape_path}")


# --- SECTION 3: FORECAST VS ACTUAL MAPE ANALYSIS ---
# Prints per-day and average MAPE for each model and stock

print("\n" + "="*60)
print("SECTION 3 - Forecast vs Actual MAPE Analysis")
print("="*60)

print(f"\n{'Stock':<12} {'Model':<8} {'MAE':>8} {'MAPE':>8}")
print("-" * 38)

for ticker, stock_name in [("BARC.L", "Barclays"), ("LLOY.L", "Lloyds"), ("HSBA.L", "HSBC")]:
    for model, stocks in FORECASTS_MAPE.items():
        act  = np.array(ACTUAL_MAPE[ticker])
        pred = np.array(stocks[ticker])
        mae  = np.mean(np.abs(pred - act))
        mape = np.mean(np.abs((pred - act) / act)) * 100
        print(f"{stock_name:<12} {model:<8} {mae:>8.2f} {mape:>7.2f}%")

print(f"\n{'Stock':<12} {'Model':<8} {'Day1':>8} {'Day2':>8} {'Day3':>8} {'Day4':>8} {'Day5':>8} {'Avg':>8}")
print("-" * 68)

for ticker, stock_name in [("BARC.L", "Barclays"), ("LLOY.L", "Lloyds"), ("HSBA.L", "HSBC")]:
    for model, stocks in FORECASTS_MAPE.items():
        act        = np.array(ACTUAL_MAPE[ticker])
        pred       = np.array(stocks[ticker])
        daily_mape = np.abs((pred - act) / act) * 100
        avg_mape   = np.mean(daily_mape)
        print(f"{stock_name:<12} {model:<8} {daily_mape[0]:>7.2f}% {daily_mape[1]:>7.2f}% {daily_mape[2]:>7.2f}% {daily_mape[3]:>7.2f}% {daily_mape[4]:>7.2f}% {avg_mape:>7.2f}%")


# --- SECTION 4: PREDICTION UNCERTAINTY CHARTS ---
# One figure per model, three subplots per stock
# Shows 5-day forecast line + CI/PI shading + value annotations
# Annotations only shown on forecast dates, not the 27 Feb anchor point

print("\n" + "="*60)
print("SECTION 4 - Building prediction uncertainty charts")
print("="*60)

for model_name, model_folder in MODELS.items():
    print(f"  Building uncertainty figure for {model_name}...")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=SUBPLOT_TITLES,
        vertical_spacing=0.18,
    )

    for row, (stock_name, stock_info) in enumerate(STOCKS.items(), start=1):
        actual_colour   = STOCK_COLOURS[stock_name]
        forecast_ticker = stock_info["forecast_ticker"]
        base_price      = BASE_PRICES[stock_name]

        # Load forecast and CI bounds
        fc_dates, fc_prices, fc_lower, fc_upper = [], [], None, None
        try:
            fc        = load_forecast(model_folder, forecast_ticker)
            fc_dates  = fc[DATE_COL].dt.strftime("%d %b").tolist()
            fc_prices = fc[FORECAST_COL].tolist()
            lower_col = get_lower_col(fc)
            upper_col = get_upper_col(fc)
            fc_lower  = fc[lower_col].tolist() if lower_col else None
            fc_upper  = fc[upper_col].tolist() if upper_col else None
        except FileNotFoundError:
            print(f"    WARNING: forecast not found for {model_name} / {forecast_ticker}")

        # CI shading polygon
        if fc_dates and fc_lower and fc_upper:
            anchor_x = ["27 Feb"] + fc_dates
            shade_x  = anchor_x + anchor_x[::-1]
            shade_y  = ([base_price] + fc_upper) + ([base_price] + fc_lower)[::-1]
            fig.add_trace(go.Scatter(
                x=shade_x,
                y=shade_y,
                fill="toself",
                fillcolor=FORECAST_FILL,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=(row == 1 and stock_name == "Barclays"),
                name="95% CI",
                legendgroup="ci",
                legendrank=999,
                hoverinfo="skip",
            ), row=row, col=1)

        # Forecast line with price annotations on forecast dates only
        if fc_dates:
            fig.add_trace(go.Scatter(
                x=["27 Feb"] + fc_dates,
                y=[base_price] + fc_prices,
                mode="lines+markers+text",
                name=stock_name,
                line=dict(color=actual_colour, width=3),
                marker=dict(size=7, color=actual_colour),
                text=[""] + [f"{p:.2f}" for p in fc_prices],
                textposition="top center",
                textfont=dict(size=9, color=actual_colour),
                showlegend=True,
                legendgroup=stock_name,
                legendrank=row,
                hovertemplate="%{y:.2f}p<extra>Forecast</extra>",
            ), row=row, col=1)

        # Upper bound annotations on forecast dates only
        if fc_dates and fc_upper:
            fig.add_trace(go.Scatter(
                x=["27 Feb"] + fc_dates,
                y=[base_price] + fc_upper,
                mode="text",
                text=[""] + [f"{p:.2f}" for p in fc_upper],
                textposition="top center",
                textfont=dict(size=8, color="rgba(140,80,0,1.0)"),
                showlegend=False,
                hoverinfo="skip",
            ), row=row, col=1)

        # Lower bound annotations on forecast dates only
        if fc_dates and fc_lower:
            fig.add_trace(go.Scatter(
                x=["27 Feb"] + fc_dates,
                y=[base_price] + fc_lower,
                mode="text",
                text=[""] + [f"{p:.2f}" for p in fc_lower],
                textposition="bottom center",
                textfont=dict(size=8, color="rgba(140,80,0,1.0)"),
                showlegend=False,
                hoverinfo="skip",
            ), row=row, col=1)

        # Scale y-axis to CI bounds so forecast movement is visible
        y_min = min(fc_lower) * 0.98 if fc_lower else None
        y_max = max(fc_upper) * 1.02 if fc_upper else None

        fig.update_yaxes(
            title_text="Price (GBX)",
            showgrid=True,
            gridcolor="#eeeeee",
            range=[y_min, y_max],
            row=row, col=1
        )

        fig.update_xaxes(
            tickangle=-45,
            showgrid=True,
            gridcolor="#eeeeee",
            row=row, col=1
        )

    fig.update_layout(
        title=dict(
            text=f"{model_name} — 5-Day Price Forecast with Prediction Intervals",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
            y=0.98,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1188,
        width=840,
        margin=dict(l=70, r=40, t=130, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
        ),
    )

    safe_name = model_name.lower().replace(" ", "_")
    out_path  = os.path.join(OUTPUT_DIR, f"uncertainty_{safe_name}.png")
    fig.write_image(out_path, scale=2)
    print(f"    Saved: {out_path}")


# --- SECTION 5: FULL TEST PERIOD COVERAGE CHARTS ---
# Full OOS test period: actual vs predicted with CI shading
# Points outside the interval are highlighted in red
# One figure per model, three subplots per stock

print("\n" + "="*60)
print("SECTION 5 - Building full test period coverage charts")
print("="*60)

for model_name, model_folder in MODELS.items():
    print(f"  Building coverage figure for {model_name}...")
    pred_suffix = PRED_SUFFIXES[model_name]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=SUBPLOT_TITLES,
        vertical_spacing=0.12,
    )

    for row, (stock_name, stock_info) in enumerate(STOCKS.items(), start=1):
        actual_colour   = STOCK_COLOURS[stock_name]
        forecast_ticker = stock_info["forecast_ticker"]

        # Load full test period predictions
        try:
            pred_df   = load_predictions(model_folder, forecast_ticker, pred_suffix)
            dates     = pred_df[DATE_COL].dt.strftime("%d %b %y").tolist()
            actual    = pred_df["Actual"].tolist()
            predicted = pred_df["Predicted"].tolist()
            lower_col = get_lower_col(pred_df)
            upper_col = get_upper_col(pred_df)
            ci_lower  = pred_df[lower_col].tolist() if lower_col else None
            ci_upper  = pred_df[upper_col].tolist() if upper_col else None
        except FileNotFoundError:
            print(f"    WARNING: predictions not found for {model_name} / {forecast_ticker}")
            continue

        # Find points where actual price falls outside the CI
        if ci_lower and ci_upper:
            outside_mask  = [(a < l or a > u) for a, l, u in zip(actual, ci_lower, ci_upper)]
            outside_dates = [d for d, m in zip(dates, outside_mask) if m]
            outside_vals  = [a for a, m in zip(actual, outside_mask) if m]
            inside_count  = sum(1 for m in outside_mask if not m)
            coverage      = inside_count / len(outside_mask) * 100

        # CI shading polygon (darker fill than Section 4)
        if ci_lower and ci_upper:
            shade_x = dates + dates[::-1]
            shade_y = ci_upper + ci_lower[::-1]
            fig.add_trace(go.Scatter(
                x=shade_x,
                y=shade_y,
                fill="toself",
                fillcolor=COVERAGE_FILL,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=(row == 1 and stock_name == "Barclays"),
                name="95% CI",
                legendgroup="ci",
                legendrank=999,
                hoverinfo="skip",
            ), row=row, col=1)

        # Predicted price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            mode="lines",
            name="Predicted",
            line=dict(color=FORECAST_COLOUR, width=1.5, dash="dash"),
            showlegend=(row == 1 and stock_name == "Barclays"),
            legendgroup="predicted",
            legendrank=998,
            hovertemplate="%{y:.2f}p<extra>Predicted</extra>",
        ), row=row, col=1)

        # Actual price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode="lines",
            name=stock_name,
            line=dict(color=actual_colour, width=2),
            showlegend=True,
            legendgroup=stock_name,
            legendrank=row,
            hovertemplate="%{y:.2f}p<extra>Actual</extra>",
        ), row=row, col=1)

        # Highlight points outside CI as larger red dots
        if ci_lower and ci_upper and outside_dates:
            fig.add_trace(go.Scatter(
                x=outside_dates,
                y=outside_vals,
                mode="markers",
                name="Outside CI",
                marker=dict(color=OUTSIDE_COLOUR, size=8, symbol="circle"),
                showlegend=(row == 1 and stock_name == "Barclays"),
                legendgroup="outside",
                legendrank=997,
                hovertemplate="%{y:.2f}p<extra>Outside CI</extra>",
            ), row=row, col=1)

        # Coverage rate annotation in top right corner of each subplot
        if ci_lower and ci_upper:
            fig.add_annotation(
                x=0.98, y=0.97,
                xref=f"x{row} domain" if row > 1 else "x domain",
                yref=f"y{row} domain" if row > 1 else "y domain",
                text=f"Coverage: {coverage:.1f}%",
                showarrow=False,
                font=dict(size=10, color="#333333"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc",
                borderwidth=1,
                xanchor="right",
                yanchor="top",
                row=row, col=1,
            )

        # Monthly x-axis ticks (approx 1 per month = 21 trading days)
        tick_vals = [dates[i] for i in range(0, len(dates), 21)]
        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_vals,
            tickangle=-45,
            showgrid=True,
            gridcolor="#eeeeee",
            row=row, col=1
        )

        fig.update_yaxes(
            title_text="Price (GBX)",
            showgrid=True,
            gridcolor="#eeeeee",
            row=row, col=1
        )

    fig.update_layout(
        title=dict(
            text=f"{model_name} — Out-of-Sample Test Period: Actual vs Predicted with 95% CI",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
            y=0.98,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1188,
        width=840,
        margin=dict(l=70, r=40, t=130, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
        ),
    )

    safe_name = model_name.lower().replace(" ", "_")
    out_path  = os.path.join(OUTPUT_DIR, f"coverage_{safe_name}.png")
    fig.write_image(out_path, scale=2)
    print(f"    Saved: {out_path}")

print("\nAll done.")