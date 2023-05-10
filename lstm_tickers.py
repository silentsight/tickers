import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from pandas.tseries.offsets import DateOffset

#Attempt at predicting with a basic lstm model that includes stock historical data and volume

def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(len(data) - sequence_length - 1):
        x.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])
    return np.array(x), np.array(y)

def predict_stock(ticker):
    # Fetch historical stock data
    df = yf.download(ticker,'2015-01-01','2023-04-30')

    # Use close prices and volume
    df = df[['Close', 'Volume']]

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Create sequences
    sequence_length = 7  # Changed from 30 to 7
    x, y = create_sequences(data_scaled, sequence_length)

    # Split the data into training and testing data
    train_length = int(len(df) * 0.7)
    x_train, x_test = x[:train_length], x[train_length:]
    y_train, y_test = y[:train_length], y[train_length:]

    # Reshape the data into three dimensions [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], df.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], df.shape[1]))

    # Build the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, df.shape[1]), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Test the model using the test data
    predictions = model.predict(x_test)

    # Reverse the scaling for the predictions by appending a column of zeros
    predictions_with_zeros = np.hstack([predictions, np.zeros((predictions.shape[0], 1))])
    predictions = scaler.inverse_transform(predictions_with_zeros)[:, 0]

    # Calculate the percentage change
    pct_change = (predictions - y_test) / y_test

    print("The average percentage change in the test set predictions is: ", np.mean(pct_change))

    # Now let's use the model to predict the next 7 days
    new_df = data_scaled[-sequence_length:]
    forecast = []

    mean_volume = data_scaled[:, 1].mean()

    for _ in range(sequence_length):
        new_df_scaled = np.reshape(new_df, (1, sequence_length, df.shape[1]))
        predicted_price = model.predict(new_df_scaled)
        forecast.append(predicted_price[0][0])
        new_day = np.array([[predicted_price[0][0], mean_volume]])
        new_df = np.vstack((new_df[1:], new_day))

    # Reverse the scaling for the forecast by appending a column of zeros
    forecast = np.array(forecast).reshape(-1, 1)
    forecast_with_zeros = np.hstack([forecast, np.zeros((forecast.shape[0], 1))])
    forecast = scaler.inverse_transform(forecast_with_zeros)[:, 0]
    print("The forecast for the next 7 days is: ", forecast)

    # Create a DataFrame for the last week of actual prices
    last_week = df['Close'].tail(sequence_length)

    # Get the day after the last day in last_week
    start_date = last_week.index[-1] + DateOffset(days=1)

    # Generate the dates for the forecast
    forecast_dates = pd.date_range(start=start_date, periods=sequence_length)

    # Create a DataFrame for the forecast using these dates
    forecast_week = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

    # Concatenate the actual and forecasted prices
    result = pd.concat([last_week, forecast_week], axis=1)
    print(result)

    # Plot the actual, training, testing and forecasted prices
    y_train_with_zeros = np.hstack([y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 1))])
    y_train_inv_transformed = scaler.inverse_transform(y_train_with_zeros)[:, 0]
    
    y_test_with_zeros = np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))])
    y_test_inv_transformed = scaler.inverse_transform(y_test_with_zeros)[:, 0]

    plt.figure(figsize=(12, 8))
    plt.plot(df.index[sequence_length:sequence_length+len(y_train)], y_train_inv_transformed, color='blue', label='Training Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], y_test_inv_transformed, color='green', label='Testing Data')
    plt.plot(df.index[sequence_length+len(y_train)+1:], predictions, color='red', label='Predicted Price')
    plt.plot(forecast_week, color='orange', label='Forecast')
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


   
