# Stock Price Prediction Website

A fully functional stock price prediction web application using Streamlit, Python, Machine Learning (LSTM & Linear Regression), and real-time live data via yfinance.

## Features
- **Real-time Data:** Fetches up-to-date historical stock prices via Yahoo Finance (`yfinance`).
- **Interactive UI:** Users can dynamically enter any ticker (e.g., AAPL) and select custom time periods.
- **Machine Learning Models:**
  - LSTM (Long Short-Term Memory) neural network.
  - Linear Regression comparison.
- **Visualizations:** Interactive charts using `Plotly` offering predictions vs historical references and confidence intervals.
- **Model Evaluation:** Real-time calculation of RMSE, MAPE, and Correlation metrics.

## Setup Instructions

### Local Development

1. Create a virtual environment (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit Application:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- Frontend: Streamlit
- Data & Preprocessing: pandas, numpy, yfinance, scikit-learn
- ML & AI: TensorFlow (Keras), scikit-learn
- Visualization: Plotly
