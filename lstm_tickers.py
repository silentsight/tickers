import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

def predict_stock(ticker):
    # Fetch historical stock data
    df = yf.download(ticker,'2015-01-01','2023-04-30')

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
    sequence_length = 30
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
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Test the model using the test data
    predictions = model.predict(x_test)

    # Reverse the scaling for the predictions
    predictions = scaler.inverse_transform(predictions)

    # Now let's use the model to predict the next 30 days
    new_df = data_scaled[-sequence_length:].copy()
    forecast = []

    for _ in range(sequence_length):
        new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], 1))
        predicted_price = model.predict(new_df_scaled)
        forecast.append(predicted_price[0])
        new_df = np.append(new_df[1:], predicted_price)

    # Reverse the scaling for the forecast
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    print("The forecast for the next 30 days is: ", forecast)

    # Create a DataFrame for the last month of actual prices
    last_month = df['Close'].tail(sequence_length)

    # Get the day after the last day in last_month
    start_date = last_month.index[-1] + pd.DateOffset(days=1)

    # Generate the dates for the forecast
    forecast_dates = pd.date_range(start=start_date, periods=sequence_length)

    # Create a DataFrame for the forecast using these dates
    forecast_month = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

    # Concatenate the actual and forecasted prices
    result = pd.concat([last_month, forecast_month], axis=1)
    print(result)

    # Plot the actual, training, testing and forecasted prices
    plt.figure(figsize=(12, 8))
    plt.plot(df.index[sequence_length:sequence_length+len(y_train)], scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Training Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Testing Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], predictions, color='red', label='Predicted Price')
    plt.plot(forecast_month, color='orange', label='Extended Prediction')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# List of stock tickers you are interested in
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Predict each stock
for ticker in tickers:
    predict_stock(ticker)
