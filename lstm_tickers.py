import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import kurtosis, skew
import seaborn as sns
import math

def predict_stock(ticker):
    # Fetch historical stock data
    df = yf.download(ticker,'2015-01-01','2023-05-10')

    # Use only close prices
    df = df[['Close']]

    # Add moving averages for trend analysis
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Add volatility for volatility analysis
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift())
    df['Volatility'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252) # annualized volatility

    # Plot the data
    df[['Close', 'MA_10', 'MA_30', 'MA_50']].plot(figsize=(10, 4), grid=True)
    plt.title(f'{ticker} Stock Price with Moving Averages')
    plt.show()

    df['Volatility'].plot(figsize=(10, 4), grid=True)
    plt.title(f'{ticker} Stock Volatility')
    plt.show()

    # Calculate skewness and kurtosis
    skewness = skew(df['Log_Returns'].dropna())
    kurt = kurtosis(df['Log_Returns'].dropna())

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    # Monte Carlo Simulation
    log_returns = np.log(1 + df['Close'].pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    t_intervals = 365
    iterations = 10
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (t_intervals, iterations)))

    S0 = df['Close'].iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    plt.figure(figsize=(10,6))
    plt.plot(price_list)
    plt.title(f'{ticker} Monte Carlo Simulation')
    plt.show()

    # Recommendation score based on volatility, skewness, kurtosis, recent performance, trend, and momentum
    volatility_threshold = 0.30  # This is an example, you can adjust this value
    skewness_threshold = 0  # This is an example, you can adjust this value
    kurtosis_threshold = 3  # This is an example, you can adjust this value
    recent_performance_threshold_1_month = 0.05  # 5% up in the past month
    recent_performance_threshold_6_month = 0.20  # 20% up in the past six months
    trend_threshold = 1.05  # Latest closing price is at least 5% above MA_50
    momentum_threshold = 0.05  # Rate of change is at least 5%

    score = 0
    if df['Volatility'].iloc[-1] < volatility_threshold:
        score += 1
    if skewness > skewness_threshold:
        score += 1
    if kurt < kurtosis_threshold:
        score += 1

    # Check recent performance (1 month and 6 months)
    recent_performance_1_month = df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1
    recent_performance_6_month = df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1
    if recent_performance_1_month > recent_performance_threshold_1_month:
        score += 1
    if recent_performance_6_month > recent_performance_threshold_6_month:
        score += 1

    # Check trend (compare latest closing price to MA_50)
    if df['Close'].iloc[-1] > trend_threshold * df['MA_50'].iloc[-1]:
        score += 1

    # Check momentum (rate of change)
    momentum = df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1
    if momentum > momentum_threshold:
        score += 1

    # Store final results
    results = pd.DataFrame({
        'Ticker': [ticker],
        'Volatility': [df['Volatility'].iloc[-1]],
        'Skewness': [skewness],
        'Kurtosis': [kurt],
        '1-Month Performance': [recent_performance_1_month],
        '6-Month Performance': [recent_performance_6_month],
        'Trend': [df['Close'].iloc[-1] / df['MA_50'].iloc[-1] - 1],
        'Momentum': [momentum],
        'Score': [score]
    })
    print(results)

    # Function to create sequences
    def create_sequences(data, sequence_length):
        x = []
        y = []
        for i in range(len(data) - sequence_length - 1):
            x.append(data[i:(i + sequence_length), 0])
            y.append(data[i + sequence_length, 0])
        return np.array(x), np.array(y)

    # Create sequences
    sequence_length = 60 # Uses the past days to forecast
    x, y = create_sequences(data_scaled, sequence_length)

    # Split the data into training and testing data
    train_length = int(len(df) * 0.7)
    x_train, x_test = x[:train_length], x[train_length:]
    y_train, y_test = y[:train_length], y[train_length:]

    # Reshape the data into three dimensions [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Test the model using the test data
    predictions = model.predict(x_test)

    # Reverse the scaling for the predictions
    predictions = scaler.inverse_transform(predictions)

    # Now let's use the model to predict the next 7 days
    new_df = data_scaled[-sequence_length:].copy()
    forecast = []

    for _ in range(7):  # Change this to 7
        new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], 1))
        predicted_price = model.predict(new_df_scaled)
        forecast.append(predicted_price[0])
        new_df = np.append(new_df[1:], predicted_price)

    # Reverse the scaling for the forecast
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    print("The forecast for the next 7 days is: ", forecast)

    # Create a DataFrame for the last week of actual prices
    last_week = df['Close'].tail(7)  # Change this to 7

    # Get the day after the last day in last_week
    start_date = last_week.index[-1] + pd.DateOffset(days=1)

    # Generate the dates for the forecast
    forecast_dates = pd.date_range(start=start_date, periods=7)  # Change this to 7

    # Create a DataFrame for the forecast using these dates
    forecast_week = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

    # Concatenate the actual and forecasted prices
    result = pd.concat([last_week, forecast_week], axis=1)

    # Calculate the percentage of change
    result['Percentage Change'] = result['Forecast'].pct_change() * 100
    print(result)

    # Plot the actual, training, testing and forecasted prices
    plt.figure(figsize=(12, 8))
    plt.plot(df.index[sequence_length:sequence_length+len(y_train)], scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Training Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Testing Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], predictions, color='red', label='Predicted Price')
    plt.plot(forecast_week, color='orange', label='Forecasted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# List of stock tickers you are interested in
tickers = ['INTA', 'NTCO', 'DVA', 'AKRO', 'CLRO']

# Predict each stock
for ticker in tickers:
    predict_stock(ticker)

