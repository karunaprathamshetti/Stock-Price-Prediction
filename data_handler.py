import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        # Convert timezone-aware datetime to timezone-naive if needed
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data_for_lstm(data, look_back=60):
    """Prepare data sequences for LSTM training"""
    prices = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    
    X, Y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        Y.append(scaled_data[i, 0])
        
    X, Y = np.array(X), np.array(Y)
    
    if len(X) > 0:
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y, scaler, scaled_data

def prepare_data_for_lr(data, look_back=60):
    """Prepare data sequences for Linear Regression training"""
    prices = data['Close'].values
    X, Y = [], []
    
    for i in range(look_back, len(prices)):
        X.append(prices[i-look_back:i])
        Y.append(prices[i])
        
    return np.array(X), np.array(Y)
