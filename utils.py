import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Since scikit-learn root_mean_squared_error was added in recent versions,
# fallback to mean_squared_error if it fails
try:
    from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
    def compute_rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    def compute_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAPE, and Correlation coefficient"""
    if len(y_true) < 2:
        return 0.0, 0.0, 0.0
        
    rmse = compute_rmse(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Avoid div-by-zero or constant sequences in correlation
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
    return rmse, mape, corr

def plot_stock_data(df, ticker):
    """Plot historical stock data"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Price'))
    fig.update_layout(
        title=f'{ticker} Historical Closing Prices', 
        xaxis_title='Date', 
        yaxis_title='Price (USD)', 
        title_font=dict(size=20, color='#f8fafc'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1')
    )
    return fig

def plot_predictions(dates, actual_prices, pred_dates, lstm_preds, lr_preds, ticker):
    """Plot historical alongside predicted data from multiple models"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual_prices, mode='lines', name='Historical Data', line=dict(color='#1f77b4')))
    
    fig.add_trace(go.Scatter(x=pred_dates, y=lstm_preds, mode='lines+markers', name='LSTM Prediction (Next 7 Days)', line=dict(color='#ff7f0e')))
    
    if lr_preds is not None:
        fig.add_trace(go.Scatter(x=pred_dates, y=lr_preds, mode='lines+markers', name='Linear Reg Prediction', line=dict(color='#2ca02c', dash='dash')))
        
    fig.update_layout(
        title=f'{ticker} Price Prediction Comparison', 
        xaxis_title='Date', 
        yaxis_title='Price (USD)', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font=dict(size=20, color='#f8fafc'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1')
    )
    return fig

def plot_confidence_intervals(dates, actual_prices, pred_dates, lstm_preds, ticker):
    """Plot LSTM predictions with a simple visual confidence interval"""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(x=dates, y=actual_prices, mode='lines', name='Historical Data', line=dict(color='#1f77b4')))
    
    # Calculate a simplified +/- 2.5% confidence interval for visualization
    upper_bound = lstm_preds * 1.025
    lower_bound = lstm_preds * 0.975
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([pred_dates, pred_dates[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval (Approx)'
    ))
    
    # LSTM Preds
    fig.add_trace(go.Scatter(x=pred_dates, y=lstm_preds, mode='lines+markers', name='LSTM Prediction', line=dict(color='#ff7f0e')))
    
    fig.update_layout(
        title=f'{ticker} Forecast with Confidence Interval', 
        xaxis_title='Date', 
        yaxis_title='Price (USD)', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font=dict(size=20, color='#f8fafc'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1')
    )
    return fig
