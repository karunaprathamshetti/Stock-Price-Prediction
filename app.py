import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Internal modules
from data_handler import fetch_data, prepare_data_for_lstm, prepare_data_for_lr
from models import build_and_train_lstm, build_and_train_lr, predict_next_days_lstm, predict_next_days_lr
from utils import calculate_metrics, plot_stock_data, plot_predictions, plot_confidence_intervals

st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

# Custom CSS for Glassmorphism, animations, and modern aesthetics
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    /* Input and Select boxes */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important;
    }

    /* Primary Button */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.2rem;
    }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Prediction")
st.markdown("Advanced AI-driven market forecasting with LSTM Neural Networks.")

# Settings Sidebar / Top row
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    ticker = st.text_input("Ticker Symbol", value="AAPL")
with col2:
    period = st.selectbox("Historical Window", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
with col3:
    epochs = st.slider("LSTM Precision (Epochs)", min_value=10, max_value=50, value=25)
with col4:
    st.markdown("<br>", unsafe_allow_html=True) # visual spacing
    predict_btn = st.button("Initialize")

if predict_btn:
    with st.spinner(f"Establishing connection... Downloading {ticker} dataset."):
        df = fetch_data(ticker, period)
        
    if df is None or len(df) < 10:
        st.error("⚠️ Insufficient data stream. Verify the ticker symbol or increase the time window.")
    else:
        st.success(f"Market data synced. {len(df)} days loaded.")
        
        # Calculate today's delta
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        delta = current_price - prev_price
        delta_pct = (delta / prev_price) * 100
        
        # Display large Top Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${current_price:.2f}", f"${delta:.2f} ({delta_pct:.2f}%)")
        m2.metric("Trading Volume", f"{int(df['Volume'].iloc[-1]):,}", "Live")
        m3.metric("Trend Strategy", "BULLISH" if delta > 0 else "BEARISH", delta_color="normal" if delta > 0 else "inverse")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create TABS
        tab1, tab2, tab3 = st.tabs(["📊 Market History", "🔮 AI Forecasting", "⚙️ Neural Validation"])
        
        with tab1:
            st.plotly_chart(plot_stock_data(df, ticker), use_container_width=True)
            with st.expander("📂 View Raw Dataset (Tabular Format)"):
                st.dataframe(df, use_container_width=True)
            
        with st.spinner("Compiling Neural Network and mapping tensors..."):
            look_back = min(60, len(df) // 4)
            if look_back < 5:
                look_back = 5
            
            X_lstm, Y_lstm, scaler, scaled_data = prepare_data_for_lstm(df, look_back)
            X_lr, Y_lr = prepare_data_for_lr(df, look_back)
            
            if len(X_lstm) < 10:
                 st.error("Historical matrix too small. Increase historical data period.")
                 st.stop()
                 
            # 80-20 Split for Validation
            split_idx = int(len(X_lstm) * 0.8)
            
            # Train Evaluation Models
            eval_lstm_model = build_and_train_lstm(X_lstm[:split_idx], Y_lstm[:split_idx], epochs=epochs)
            eval_lr_model = build_and_train_lr(X_lr[:split_idx], Y_lr[:split_idx])
            
            # Predictions on Test set
            lstm_test_preds = []
            for i in range(split_idx, len(X_lstm)):
                pred = eval_lstm_model.predict(X_lstm[i].reshape(1, look_back, 1), verbose=0)
                lstm_test_preds.append(pred[0, 0])
                
            lstm_test_preds = scaler.inverse_transform(np.array(lstm_test_preds).reshape(-1, 1)).flatten()
            y_test_true = scaler.inverse_transform(Y_lstm[split_idx:].reshape(-1, 1)).flatten()
            lr_test_preds = eval_lr_model.predict(X_lr[split_idx:])
            
            # Metrics Calculation
            lstm_rmse, lstm_mape, lstm_corr = calculate_metrics(y_test_true, lstm_test_preds)
            lr_rmse, lr_mape, lr_corr = calculate_metrics(y_test_true, lr_test_preds)
            
            # FINAL Future Model
            lstm_model = build_and_train_lstm(X_lstm, Y_lstm, epochs=epochs)
            lr_model = build_and_train_lr(X_lr, Y_lr)
            
            # 7-Day Forecast inference
            last_lstm_seq = scaled_data[-look_back:]
            last_lr_seq = df['Close'].values[-look_back:]
            
            next_7_days_lstm = predict_next_days_lstm(lstm_model, last_lstm_seq, scaler, days=7)
            next_7_days_lr = predict_next_days_lr(lr_model, last_lr_seq, days=7)
            
            last_date = df['Date'].iloc[-1]
            valid_dates = []
            current_date = last_date
            while len(valid_dates) < 7:
                current_date += datetime.timedelta(days=1)
                if current_date.weekday() < 5:
                    valid_dates.append(current_date)
                    
        with tab2:
            st.subheader("7-Day Trajectory Generation")
            hist_dates = df['Date'].tail(90)
            hist_prices = df['Close'].tail(90)
            
            p_fig = plot_predictions(hist_dates, hist_prices, valid_dates, next_7_days_lstm, next_7_days_lr, ticker)
            st.plotly_chart(p_fig, use_container_width=True)
            
            st.subheader("Statistical Confidence Bounds")
            c_fig = plot_confidence_intervals(hist_dates, hist_prices, valid_dates, next_7_days_lstm, ticker)
            st.plotly_chart(c_fig, use_container_width=True)
            
            st.markdown("#### Projected Target Prices")
            pred_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in valid_dates],
                'LSTM Target': [f"${x:.2f}" for x in next_7_days_lstm],
                'Linear Reg Target': [f"${x:.2f}" for x in next_7_days_lr]
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
        with tab3:
            st.subheader("Model Validation Accuracy")
            st.info("Metrics calculated on internal 20% holdout test data.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🧠 LSTM Neural Net")
                st.metric("Root Mean Sq Error (RMSE)", f"{lstm_rmse:.2f}")
                st.metric("Mean Abs Pct Error", f"{lstm_mape:.2%}")
                st.metric("Trend Correlation", f"{lstm_corr:.2f}")
            with c2:
                st.markdown("### 📉 Linear Regression")
                st.metric("Root Mean Sq Error (RMSE)", f"{lr_rmse:.2f}")
                st.metric("Mean Abs Pct Error", f"{lr_mape:.2%}")
                st.metric("Trend Correlation", f"{lr_corr:.2f}")
