# debug_column_order.py
import pandas as pd
import numpy as np
import os

LAGS = [1, 2, 3, 5]

df = pd.read_csv(
    "saved_data/BARC_L_20210228_20260228.csv",
    index_col=0, parse_dates=True
).sort_index()

price_col = 'Adj Close'

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

df_feat = create_enhanced_features(df, price_col=price_col)

feature_columns = [c for c in df_feat.columns if c.startswith((
    'Ret_Lag', 'SMA', 'RSI', 'BB', 'Volatility', 'Volume', 'ATR'))]

print("Column order from rf_analysis.py:")
for i, col in enumerate(feature_columns):
    print(f"  [{i}] {col}")