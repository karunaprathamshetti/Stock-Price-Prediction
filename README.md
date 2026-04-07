# Stock Price Prediction App

Hey! This is a simple web app I built to predict stock prices using Python and Machine Learning. It uses Streamlit for the frontend so everything runs directly in the browser. 

The core idea is pretty straightforward: it grabs the latest historical stock data from Yahoo Finance and feeds it into an LSTM (Long Short-Term Memory) neural network to forecast the closing prices for the next 7 days. I also added a standard Linear Regression model alongside it just to compare how a basic algorithm performs against the deep learning model.

### What it does:
* Lets you enter any stock ticker (like AAPL or TSLA) and pull historical data on the fly.
* Shows you interactive charts (built with Plotly) so you can zoom in on past trends.
* Trains an LSTM network on your machine to guess the next 7 days.
* Spits out metrics (RMSE, MAPE) so you can see how inaccurate/accurate the models actually were on the test data before trusting the future prediction.

### How to run it locally

If you want to run this on your own machine instead of the cloud, here's how:

1. Clone this repo to your computer.
2. (Optional but recommended) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows run: venv\Scripts\activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Streamlit server:
   ```bash
   python -m streamlit run app.py
   ```
   
The app should automatically pop up in your browser at `localhost:8501`. 

### Cloud Deployment
I've already set it up so it can be deployed for free on Streamlit Community Cloud. The `.python-version` file ensures that Streamlit uses Python 3.11, which avoids some nasty compilation errors with TensorFlow. Just connect your GitHub to Streamlit, point it to `app.py`, and hit deploy.

***Note:** Just a quick disclaimer that this is a personal project for learning ML. Please don't use this as actual financial advice to trade real money.*