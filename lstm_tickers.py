import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

def load_data(ticker):
    # Fetch historical stock data
    df = yf.download(ticker, '2015-01-01', '2023-05-10')
    return df

def visualize_data(df):
    # Plot historical stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'])
    plt.title('Historical Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def calculate_moving_averages(df):
    # Calculate moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

def plot_moving_averages(df):
    # Plot moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Price')
    plt.plot(df.index, df['MA_20'], label='MA 20')
    plt.plot(df.index, df['MA_50'], label='MA 50')
    plt.title('Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_volatility(df):
    # Calculate volatility
    df['Log_Return'] = np.log(df['Close']).diff()
    df['Volatility'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)

def plot_volatility(df):
    # Plot volatility
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Volatility'])
    plt.title('Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

def run_monte_carlo_simulation(df):
    # Run Monte Carlo simulation
    log_returns = np.log(1 + df['Close'].pct_change())
    mean = log_returns.mean()
    std = log_returns.std()

    num_simulations = 1000
    num_days = 7
    simulation_df = pd.DataFrame()

    for i in range(num_simulations):
        prices = []
        price = df['Close'].iloc[-1]
        for j in range(num_days):
            price = price * (1 + np.random.normal(mean, std))
            prices.append(price)
        simulation_df[i] = prices

    # Plot Monte Carlo simulation results
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_df)
    plt.title('Monte Carlo Simulation')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def calculate_skewness(df):
    # Calculate skewness of returns
    returns = df['Close'].pct_change().dropna()
    skewness = skew(returns)
    return skewness

def calculate_kurtosis(df):
    # Calculate kurtosis of returns
    returns = df['Close'].pct_change().dropna()
    kurt = kurtosis(returns)
    return kurt

def calculate_recommendation_score(skewness, kurtosis):
    # Calculate recommendation score based on skewness and kurtosis
    score = 0
    if skewness < 0:
        score -= 1
    elif skewness > 0:
        score += 1

    if kurtosis < 0:
        score -= 1
    elif kurtosis > 0:
        score += 1

    return score

def predict_stock(ticker):
    # Fetch historical stock data
    df = load_data(ticker)

    # Visualize the data
    visualize_data(df)

    # Use only close prices
    df = df[['Close']]

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Function to create sequences
    def create_sequences(data, sequence_length):
        x = []
        y = []
        for i in range(len(data) - sequence_length - 1):
            x.append(data[i:(i + sequence_length), 0])
            y.append(data[i + sequence_length, 0])
        return np.array(x), np.array(y)

    # Create sequences
    sequence_length = 60  # Uses the past days to forecast
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

    for _ in range(7):
        new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], 1))
        predicted_price = model.predict(new_df_scaled)
        forecast.append(predicted_price[0])
        new_df = np.append(new_df[1:], predicted_price)

    # Reverse the scaling for the forecast
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    print("The forecast for the next 7 days is: ", forecast)

    # Create a DataFrame for the last week of actual prices
    last_week = df['Close'].tail(7)

    # Get the day after the last day in last_week
    start_date = last_week.index[-1] + pd.DateOffset(days=1)

    # Generate the dates for the forecast
    forecast_dates = pd.date_range(start=start_date, periods=7)

    # Create a DataFrame for
    forecast_week = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

    # Concatenate the actual and forecasted prices
    result = pd.concat([last_week, forecast_week], axis=1)

    # Calculate the percentage change
    result['Percentage Change'] = result['Forecast'].pct_change() * 100
    print(result)

    # Calculate skewness and kurtosis
    returns = df['Close'].pct_change().dropna()
    skewness = skew(returns)
    kurt = kurtosis(returns)

    # Calculate the recommendation score
    score = calculate_recommendation_score(skewness, kurt)
    print("Recommendation Score:", score)

    # Plot the actual, training, testing, and forecasted prices
    plt.figure(figsize=(12, 8))
    plt.plot(df.index[sequence_length:sequence_length+len(y_train)], scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Training Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Testing Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], predictions, color='red', label='Predicted Price')
    plt.plot(forecast_week.index, forecast_week['Forecast'], color='orange', label='Extended Prediction')
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
  
