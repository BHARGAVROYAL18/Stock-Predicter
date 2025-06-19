import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

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
    {"label": "1 Day", "period_days": 2, "predict_steps": 12, "note": "Predicting the next 12 hours.", "interval": "30m", "min_points": 20},
    {"label": "1 Week", "period_days": 30, "predict_steps": 3, "note": "Predicting the next 3 days.", "interval": "1d", "min_points": 5},
    {"label": "1 Month", "period_days": 60, "predict_steps": 15, "note": "Predicting the next 15 days.", "interval": "1d", "min_points": 20},
    {"label": "6 Months", "period_days": 365, "predict_steps": 21, "note": "Predicting the next 1 month.", "interval": "1d", "min_points": 20},
    {"label": "1 Year", "period_days": 1825, "predict_steps": 126, "note": "Predicting the next 6 months.", "interval": "1d", "min_points": 20},
]

def day_suffix(day):
    if 11 <= day <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

def format_date(dt):
    return f"{dt.day}{day_suffix(dt.day)} {dt.strftime('%b')}"

def format_datetime(dt):
    return f"{dt.day}{day_suffix(dt.day)} {dt.strftime('%b %H:%M')}"

for config in graph_configs:
    label = config["label"]
    period_days = config["period_days"]
    predict_steps = config["predict_steps"]
    note = config["note"]
    interval = config["interval"]
    min_points = config["min_points"]

    # Download real-time data for the period
    period_start = end - timedelta(days=period_days)
    real_data = yf.download(stock, period_start, end, interval=interval)
    fallback_used = False
    # Fallback to daily if intraday not available or too little data
    if (real_data is None or len(real_data) < min_points) and interval != '1d':
        real_data = yf.download(stock, period_start, end, interval='1d')
        fallback_used = True
    if real_data is None or len(real_data) < min_points:
        st.warning(f"Not enough data for {label} graph, even with daily fallback.")
        continue
    close_prices = real_data['Close'].values
    # Use as much as available, at least min_points, at most 100
    n_hist = min(len(close_prices), 100)
    close_hist = close_prices[-n_hist:]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_hist.reshape(-1,1))
    last_seq = scaled[-n_hist:]
    # If less than 100, pad with the first value
    if n_hist < 100:
        pad = np.full((100-n_hist, 1), scaled[0,0])
        last_seq = np.vstack([pad, last_seq])
    preds = predict_future(model, last_seq, predict_steps, scaler)

    # For x-axis: use actual date/time labels
    hist_index = real_data.index[-n_hist:]
    if interval == '30m':
        # For intraday, next steps are 30m intervals
        last_time = hist_index[-1]
        pred_index = [last_time + pd.Timedelta(minutes=30*(i+1)) for i in range(predict_steps)]
        x_hist = hist_index
        x_pred = pred_index
        date_fmt = '%Y-%m-%d %H:%M'
    else:
        # For daily, next steps are days
        last_date = hist_index[-1]
        pred_index = [last_date + pd.Timedelta(days=i+1) for i in range(predict_steps)]
        x_hist = hist_index
        x_pred = pred_index
        date_fmt = '%Y-%m-%d'

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x_hist, close_hist, color='#1976d2', linewidth=2, label='Real-Time Close Price')
    ax.plot(x_pred, preds, color='#ff9800', linewidth=2, label=f'Predicted ({note})')
    ax.scatter([x_hist[-1]], [close_hist[-1]], color='green', s=40, label='Prediction Start')
    ax.scatter([x_pred[-1]], [preds[-1]], color='red', s=40, label='Prediction End')
    ax.set_xlabel('Date/Time', fontsize=10)
    ax.set_ylabel('Close Price', fontsize=10)
    ax.set_title(f'{label} Graph: Real-Time vs Predicted', fontsize=12, color='#1a237e')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    # Format x-axis with custom formatter
    if interval == '30m':
        def intraday_fmt(x, pos=None):
            try:
                dt = mdates.num2date(x)
                return format_datetime(dt)
            except Exception:
                return ''
        ax.xaxis.set_major_formatter(FuncFormatter(intraday_fmt))
    else:
        def daily_fmt(x, pos=None):
            try:
                dt = mdates.num2date(x)
                return format_date(dt)
            except Exception:
                return ''
        ax.xaxis.set_major_formatter(FuncFormatter(daily_fmt))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    st.subheader(f'🔮 {label} Graph')
    st.markdown(f"<span style='color:#1976d2;font-size:14px'>{note}</span>", unsafe_allow_html=True)
    if fallback_used:
        st.info("Intraday data not available, using daily data as fallback.")
    st.pyplot(fig)
    st.markdown("---")

st.markdown("<center style='color:#1976d2; font-size:18px;'>Made with ❤️ using Streamlit & Keras</center>", unsafe_allow_html=True)