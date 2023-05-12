import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import kurtosis, skew
import seaborn as sns
import math
from ta.momentum import rsi
from ta.trend import macd

SEQUENCE = 60
PERIOD = 7


def predict_stock(ticker):
    # Fetch historical stock data
    df = yf.download(ticker, '2015-01-01', '2023-05-12')

    # Use only close prices
    df = df[['Close']]

    # Add moving averages for trend analysis
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Add volatility for volatility analysis
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift())
    df['Volatility'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252)  # annualized volatility

    # Calculate RSI and MACD
    df['RSI'] = rsi(df['Close'])
    df['MACD'] = macd(df['Close'])

    # Plot the data
    df[['Close', 'MA_10', 'MA_30', 'MA_50']].plot(figsize=(10, 4), grid=True)
    plt.title(f'{ticker} Stock Price with Moving Averages')
    plt.show()

    df['Volatility'].plot(figsize=(10, 4), grid=True)
    plt.title(f'{ticker} Stock Volatility')
    plt.show()

    df['RSI'].plot(figsize=(10, 4), grid=True)
    plt.title(f'{ticker} RSI')
    plt.show()

    df['MACD'].plot(figsize=(10, 4), grid=True)
    plt.title(f'{ticker} MACD')
    plt.show()

    # Calculate skewness and kurtosis
    skewness = skew(df['Log_Returns'].dropna())
    kurt = kurtosis(df['Log_Returns'].dropna())

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

    plt.figure(figsize=(10, 6))
    plt.plot(price_list)
    plt.title(f'{ticker} Monte Carlo Simulation')
    plt.show
    # Recommendation score based on volatility, skewness, kurtosis, recent performance, trend, momentum, RSI, and MACD
    volatility_threshold = 0.30  # This is an example, you can adjust this value
    skewness_threshold = 0  # This is an example, you can adjust this value
    kurtosis_threshold = 3  # This is an example, you can adjust this value
    recent_performance_threshold_1_month = 0.05  # 5% up in the past month
    recent_performance_threshold_6_month = 0.20  # 20% up in the past six months
    trend_threshold = 1.05  # Latest closing price is at least 5% above MA_50
    momentum_threshold = 0.05  # Rate of change is at least 5%
    rsi_threshold = 50  # RSI threshold value
    macd_threshold = 0  # MACD threshold value

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

    # Check RSI
    if df['RSI'].iloc[-1] > rsi_threshold:
        score += 1

    # Check MACD
    if df['MACD'].iloc[-1] > macd_threshold:
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
        'RSI': [df['RSI'].iloc[-1]],
        'MACD': [df['MACD'].iloc[-1]],
        'Score': [score]
    })
    print(results)

    # Drop the rows with missing values
    df = df.dropna()

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Add another scaler for 'Close' prices only
    close_scaler = MinMaxScaler()
    df[['Close']] = close_scaler.fit_transform(df[['Close']])
    # Function to create sequences
    def create_sequences(data, sequence_length):
        x = []
        y = []
        for i in range(len(data) - sequence_length - 1):
            x.append(data[i:(i + sequence_length), :])
            y.append(data[i + sequence_length, 0])  # Predict the next 'Close' price
        return np.array(x), np.array(y).reshape(-1, 1)

    # Create sequences
    sequence_length = SEQUENCE  # Adjust sequence length
    x, y = create_sequences(data_scaled, sequence_length)

    # Split the data into training and testing data
    train_length = int(len(df) * 0.7)
    x_train, x_test = x[:train_length], x[train_length:]
    y_train, y_test = y[:train_length], y[train_length:]

    # Build the CNN-GRU model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 8)),
        MaxPooling1D(pool_size=2),
        GRU(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        GRU(100, activation='relu'),
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
    predictions = close_scaler.inverse_transform(predictions)

    # Now let's use the model to predict the next 14 days
    new_df = data_scaled[-sequence_length:].copy()  # shape: (sequence_length, 8)
    forecast = []

    df['Close'] = close_scaler.inverse_transform(df[['Close']])

    for _ in range(PERIOD):  # Change this to 14
        new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], new_df.shape[1]))
        predicted_price = model.predict(new_df_scaled)  # shape: (1, 1)
        # Propagate the last features
        last_features = new_df[-1, 1:]
        new_prediction = np.concatenate([predicted_price, last_features.reshape(1, -1)], axis=1)  # shape: (1, 8)
        new_df = np.concatenate([new_df[1:], new_prediction])  # shape: (sequence_length, 8)

        forecast.append(predicted_price[0])

    # Reverse the scaling for the forecast
    forecast = close_scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    print("The forecast for the next 14 days is: ", forecast)

    # Create a DataFrame for the last week of actual prices
    last_week = df['Close'].tail(PERIOD)  # Change this to 14

    # Get the day after the last day in last_week
    start_date = last_week.index[-1] + pd.DateOffset(days=1)

    # Generate the dates for the forecast
    forecast_dates = pd.date_range(start=start_date, periods=PERIOD)  # Change this to 14

    # Create a DataFrame for the forecast using these dates
    forecast_week = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

    # Concatenate the actual and forecasted prices
    result = pd.concat([last_week, forecast_week], axis=1)

    # Calculate the percentage of change
    result['Percentage Change'] = result['Forecast'].pct_change() * 100
    print(result)

    # Plot the actual, training, testing, and forecasted prices
    plt.figure(figsize=(12, 8))
    plt.plot(df.index[sequence_length:sequence_length + len(y_train)],
             close_scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Training Data')
    plt.plot(df.index[sequence_length + len(y_train) + 1:],
             close_scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Testing Data')
    plt.plot(df.index[sequence_length + len(y_train) + 1:], predictions, color='red', label='Predicted Price')
    plt.plot(forecast_week, color='orange', label='Forecasted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# List of stock tickers you are interested in
tickers = ['BTC-USD', 'INTA', 'NTCO', 'DVA', 'AKRO', 'CLRO']

# Predict each stock
for ticker in tickers:
    predict_stock(ticker)
