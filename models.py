import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression

def build_and_train_lstm(X_train, y_train, epochs=20, batch_size=32):
    """Build and train an LSTM model"""
    model = Sequential()
    
    # Check if there is enough data
    if len(X_train) == 0:
        return None
        
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    return model

def build_and_train_lr(X_train, y_train):
    """Build and train a Linear Regression model"""
    model = LinearRegression()
    if len(X_train) > 0:
        model.fit(X_train, y_train)
    return model

def predict_next_days_lstm(model, last_sequence, scaler, days=7):
    """Predict the next N days using trained LSTM model"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
        predictions.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred[0, 0])
        
    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def predict_next_days_lr(model, last_sequence, days=7):
    """Predict the next N days using trained Linear Regression model"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, -1))[0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)
        
    return np.array(predictions)
