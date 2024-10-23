# Required libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import tensorflow.compat.v2 as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Streamlit configuration
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# App title and image
st.title('Stock Market Forecasting App')
st.subheader('This app forecasts the stock price of the selected company.')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Sidebar inputs
st.sidebar.header('Select the parameters below')
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Data visualization
st.header('Data Visualization')
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# Select column for forecasting
column = st.selectbox('Select column for forecasting', data.columns[1:])
data = data[['Date', column]]

# Stationarity check (ADF test)
st.header('Is data Stationary?')
adf_p_value = adfuller(data[column])[1]
st.write(f"P-value of ADF Test: {adf_p_value}")
if adf_p_value < 0.05:
    st.write("The data is stationary.")
else:
    st.write("The data is not stationary.")

# Decompose data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals'))

# Model selection
models = ['SARIMA', 'Random Forest', 'LSTM', 'Prophet']
selected_model = st.sidebar.selectbox('Select model for forecasting', models)

# SARIMA Model
if selected_model == 'SARIMA':
    p = st.slider('p', 0, 5, 2)
    d = st.slider('d', 0, 5, 1)
    q = st.slider('q', 0, 5, 2)
    seasonal_order = st.number_input('Seasonal order (p)', 0, 24, 12)
    
    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model_fit = model.fit()
    st.write(model_fit.summary())

    forecast_period = st.number_input('Forecast period (days)', 1, 365, 10)
    predictions = model_fit.get_prediction(start=len(data), end=len(data) + forecast_period)
    forecast = predictions.predicted_mean
    forecast.index = pd.date_range(start=end_date, periods=len(forecast), freq='D')
    st.write(forecast)

    # Plot forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Predicted'))
    st.plotly_chart(fig)

# Random Forest Model
elif selected_model == 'Random Forest':
    st.write('Random Forest Regression')

    # Data preparation and model training
    train_size = int(len(data) * 0.8)
    train_X, train_y = data['Date'][:train_size], data[column][:train_size]
    test_X, test_y = data['Date'][train_size:], data[column][train_size:]
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train_X.values.reshape(-1, 1), train_y.values)

    # Prediction and evaluation
    predictions = rf_model.predict(test_X.values.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    st.write(f"RMSE: {rmse}")

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_X, y=train_y, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_X, y=predictions, mode='lines', name='Predicted'))
    st.plotly_chart(fig)

# LSTM Model
elif selected_model == 'LSTM':
    st.write('LSTM Model')

    # Data scaling and splitting
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length])
            y.append(dataset[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = st.slider('Sequence length', 1, 30, 10)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)

    # LSTM model building and training
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_X, train_y, epochs=10, batch_size=16)

    # Prediction and evaluation
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual'))
    st.plotly_chart(fig)

# Prophet Model
elif selected_model == 'Prophet':
    prophet_data = data.rename(columns={'Date': 'ds', column: 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)
    
    fig = prophet_model.plot(forecast)
    st.pyplot(fig)

st.write(f"Model selected: {selected_model}")


# redirect URLs
github_redirect_url = "https://github.com/AliAkber12/Ali-Akber"
kaggle_redirect_url = "https://www.kaggle.com/"
linkedin_redirect_url = "https://www.linkedin.com/in/ali-akber-chandio-480344329/"

# Footer with social media icons
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        color: #000000;
        text-align: center;
        padding: 10px;
    }
    .footer img {
        margin: 0 10px;
        vertical-align: middle;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="footer">
        Made by Ali Akber Chandio ❤️
        <a href="{github_redirect_url}" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30" alt="GitHub">
        </a>
        <a href="{kaggle_redirect_url}" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original-wordmark.svg" width="30" height="30" alt="Kaggle">
        </a>
        <a href="{linkedin_redirect_url}" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" height="30" alt="LinkedIn">
        </a>
    </div>
    """, 
    unsafe_allow_html=True
)
