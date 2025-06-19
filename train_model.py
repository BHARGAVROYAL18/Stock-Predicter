import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Step 1: Download stock data
stock = "GOOG"
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
data = yf.download(stock, start=start, end=end)
close_data = data[['Close']]

# Step 2: Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# Step 3: Create sequences
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])
x_data, y_data = np.array(x_data), np.array(y_data)

# Step 4: Build and compile the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_data.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(x_data, y_data, epochs=5, batch_size=32, verbose=1)

# Step 6: Save the model
model.save("Latest_stock_price_model.keras")
print("✅ Model saved as 'Latest_stock_price_model.keras'")
