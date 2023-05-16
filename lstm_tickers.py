import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Conv1D, MaxPooling1D, Dropout, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from ta.momentum import rsi
from ta.trend import macd
from pandas.tseries.offsets import CustomBusinessHour
import matplotlib
from datetime import date

# Create a custom business hour object
cbh = CustomBusinessHour(start='09:30', end='16:00')

# Hyperparameters
SEQUENCE_LENGTH = 60
PERIOD = 24
LSTM_UNITS = 100
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
BATCH_SIZE = 32
NUM_CONV_LAYERS = 1
NUM_FILTERS = 64
KERNEL_SIZE = 3
NUM_POOLING_LAYERS = 1
POOL_SIZE = 2
HISTORICAL_INTERVAL = "1h"
START_DATE = "2023-01-01"
END_DATE = "2023-05-15"

# List of stock tickers you are interested in
tickers = ['ARRY', 'INTA', 'NTCO', 'DVA', 'AKRO', 'CLRO', 'APP', 'MNDY', 'TGOPY']

def get_stock_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, interval=HISTORICAL_INTERVAL)
    if df.empty:
        raise ValueError(f"No historical stock data found for {ticker}")
    return df

def preprocess_data(df):
    df = df[['Close']]
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift())
    df['Volatility'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252)
    df['RSI'] = rsi(df['Close'])
    df['MACD'] = macd(df['Close'])
    return df.dropna()

def scale_data(df):
    scaler = MinMaxScaler()
    close_scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df['Close'] = close_scaler.fit_transform(df[['Close']])
    return df_scaled, close_scaler

def split_data(df_scaled):
    sequence_length = SEQUENCE_LENGTH
    x, y = create_sequences(df_scaled, sequence_length)
    train_length = int(len(df_scaled) * 0.7)
    x_train, x_test = x[:train_length], x[train_length:]
    y_train, y_test = y[:train_length], y[train_length:]
    return x_train, x_test, y_train, y_test

def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(len(data) - sequence_length - 1):
        x.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length, 0])
    return np.array(x), np.array(y).reshape(-1, 1)

def build_model(sequence_length):
    model = Sequential()
    for _ in range(NUM_CONV_LAYERS):
        model.add(Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu',
                         input_shape=(sequence_length, 8)))
        model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(GRU(LSTM_UNITS, activation='relu', return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(GRU(LSTM_UNITS, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def predict_future(df_scaled, model, close_scaler, sequence_length, df):
    new_df = df_scaled[-sequence_length:].copy()
    forecast = []
    for _ in range(PERIOD):
        new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], new_df.shape[1]))
        predicted_price = model.predict(new_df_scaled)
        last_features = new_df[-1, 1:]
        new_prediction = np.concatenate([predicted_price, last_features.reshape(1, -1)], axis=1)
        new_df = np.concatenate([new_df[1:], new_prediction])
        forecast.append(predicted_price[0])
    forecast = close_scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Create a date range for the forecasted data
    last_date = df.index[-1]
    cbh = CustomBusinessHour(start='09:30', end='16:00')
    forecast_index = [last_date + cbh * i for i in range(1, PERIOD + 1)]
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    return forecast_df

def plot_data(df, y_train, y_test, predictions, forecast_df, close_scaler, sequence_length, ticker):
    plt.figure(figsize=(12, 8))
    plt.plot(df.index[sequence_length:sequence_length + len(y_train)],
             close_scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Training Data')
    plt.plot(df.index[sequence_length + len(y_train) + 1:],
             close_scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Testing Data')
    plt.plot(df.index[sequence_length + len(y_train) + 1:], predictions, color='red', label='Predicted Price')
    plt.plot(forecast_df.index, forecast_df['Forecast'], color='orange', label='Forecasted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'{date.today()}_{ticker}_prediction.png')

def predict_stock(ticker):
    try:
        df = get_stock_data(ticker)
        df = preprocess_data(df)
        df_scaled, close_scaler = scale_data(df)
        x_train, x_test, y_train, y_test = split_data(df_scaled)
        model = build_model(SEQUENCE_LENGTH)
        model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        predictions = model.predict(x_test)
        predictions = close_scaler.inverse_transform(predictions)
        forecast_df = predict_future(df_scaled, model, close_scaler, SEQUENCE_LENGTH, df)
        last_real_price = close_scaler.inverse_transform([[df_scaled[-1, 0]]])[0, 0]
        forecast_df = pd.concat([pd.DataFrame([last_real_price], columns=['Forecast'], index=[df.index[-1]]), forecast_df])
        plot_data(df, y_train, y_test, predictions, forecast_df, close_scaler, SEQUENCE_LENGTH, ticker)
        forecast_df['Ticker'] = ticker
        return forecast_df
    except Exception as e:
        print("An error occurred during stock prediction:", str(e))
        return pd.DataFrame()

all_forecasts = []
for ticker in tickers:
    forecast_df = predict_stock(ticker)
    all_forecasts.append(forecast_df)

all_forecasts_df = pd.concat(all_forecasts)
all_forecasts_df = all_forecasts_df.pivot(columns='Ticker')

# Create a new DataFrame for the percentage change
percentage_change_df = all_forecasts_df.pct_change() * 100

# Rename the columns in the percentage change DataFrame
percentage_change_df.columns = [f'{col}_Percentage Change' for col in percentage_change_df.columns]

# Concatenate the original DataFrame and the percentage change DataFrame
final_df = pd.concat([all_forecasts_df, percentage_change_df], axis=1)

print(final_df)

final_df.to_csv(f'{date.today()}_predicted_stocks.csv')




