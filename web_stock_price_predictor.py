import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Set Streamlit page config for a modern look
st.set_page_config(page_title="Stock Price Predictor App", page_icon="📈", layout="wide")

# Custom CSS for improved UI
st.markdown("""
    <style>
        body, .stApp { background-color: #f7f9fa !important; color: #222 !important; }
        .main { background-color: #f7f9fa !important; }
        h1, h2, h3, h4, h5, h6 { color: #1a237e !important; }
        .stTextInput > div > div > input { background-color: #fff !important; color: #1a237e !important; border-radius: 8px; border: 1px solid #90caf9; }
        .stDataFrame { background-color: #fff !important; border-radius: 8px; }
        .stButton > button { background-color: #1976d2 !important; color: #fff !important; border-radius: 8px; }
        .stSubheader { color: #1976d2 !important; }
        .stMarkdown { color: #374151 !important; }
        .block-container { padding: 2rem 2rem 2rem 2rem; }
        hr { border: 1px solid #90caf9; margin: 2rem 0; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Predictor App")
st.markdown("""
Welcome to the **Stock Price Predictor**! Enter a stock ticker below to view historical data and future price predictions using a deep learning model.
""")

# Input section
with st.container():
    st.markdown("---")
    stock = st.text_input("🔎 Enter the Stock Ticker (e.g., GOOG, AAPL, MSFT)", "GOOG")
    st.markdown("---")

end = datetime.now()
start = datetime(end.year-2, end.month, end.day)  # 2 years for context

google_data = yf.download(stock, start, end, interval='1d')

model = load_model("Latest_stock_price_model.keras")

# Data section
st.subheader("📊 Stock Data Table")
st.dataframe(google_data.tail(20), use_container_width=True)
st.markdown("---")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Helper for multi-step prediction
def predict_future(model, last_sequence, n_steps, scaler):
    future_preds = []
    current_seq = last_sequence.copy()
    for _ in range(n_steps):
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], 1), verbose=0)
        future_preds.append(pred[0,0])
        current_seq = np.append(current_seq[1:], pred, axis=0)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()

# Graph configs: period, prediction window, note, yfinance interval
graph_configs = [
    {"label": "1 Day", "period_days": 2, "predict_steps": 12, "note": "Predicting the next 12 hours.", "interval": "30m"},
    {"label": "1 Week", "period_days": 14, "predict_steps": 3, "note": "Predicting the next 3 days.", "interval": "1d"},
    {"label": "1 Month", "period_days": 60, "predict_steps": 15, "note": "Predicting the next 15 days.", "interval": "1d"},
    {"label": "6 Months", "period_days": 365, "predict_steps": 21, "note": "Predicting the next 1 month.", "interval": "1d"},
    {"label": "1 Year", "period_days": 730, "predict_steps": 126, "note": "Predicting the next 6 months.", "interval": "1d"},
]

for config in graph_configs:
    label = config["label"]
    period_days = config["period_days"]
    predict_steps = config["predict_steps"]
    note = config["note"]
    interval = config["interval"]

    # Download real-time data for the period
    period_start = end - timedelta(days=period_days)
    real_data = yf.download(stock, period_start, end, interval=interval)
    if len(real_data) < 100:
        st.warning(f"Not enough data for {label} graph.")
        continue
    close_prices = real_data['Close'].values
    # Prepare last 100 for prediction
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_prices[-100:].reshape(-1,1))
    last_100 = scaled[-100:]
    preds = predict_future(model, last_100, predict_steps, scaler)

    # For x-axis
    x_hist = np.arange(len(close_prices))
    x_pred = np.arange(len(close_prices), len(close_prices)+predict_steps)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x_hist, close_prices, color='#1976d2', linewidth=2, label='Real-Time Close Price')
    ax.plot(x_pred, preds, color='#ff9800', linewidth=2, label=f'Predicted ({note})')
    ax.scatter([x_hist[-1]], [close_prices[-1]], color='green', s=40, label='Prediction Start')
    ax.scatter([x_pred[-1]], [preds[-1]], color='red', s=40, label='Prediction End')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Close Price', fontsize=10)
    ax.set_title(f'{label} Graph: Real-Time vs Predicted', fontsize=12, color='#1a237e')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    st.subheader(f'🔮 {label} Graph')
    st.markdown(f"<span style='color:#1976d2;font-size:14px'>{note}</span>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("---")

st.markdown("<center style='color:#1976d2; font-size:18px;'>Made with ❤️ using Streamlit & Keras</center>", unsafe_allow_html=True)