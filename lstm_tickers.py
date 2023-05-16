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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from ta.momentum import rsi
from ta.trend import macd
from bs4 import BeautifulSoup
import requests
import datetime


# List of stock tickers you are interested in
tickers = ['ARRY']

SEQUENCE = 60
PERIOD = 7

# Adjustable hyperparameters
SEQUENCE_LENGTH = 60  # Increase to capture longer-term patterns
PERIOD = 7  # Adjust to determine the number of days to predict
LSTM_UNITS = 100  # Increase to capture more complex patterns
DROPOUT_RATE = 0.2  # Adjust to regularize the model and prevent overfitting
LEARNING_RATE = 0.0001  # Experiment with different learning rates
NUM_EPOCHS = 50  # Increase to allow the model to train for more iterations
BATCH_SIZE = 32  # Adjust based on available memory resources
NUM_CONV_LAYERS = 1  # Increase to capture more complex patterns
NUM_FILTERS = 64  # Adjust to control model's capacity to capture features
KERNEL_SIZE = 3  # Experiment with different kernel sizes
NUM_POOLING_LAYERS = 1  # Increase to downsample feature maps
POOL_SIZE = 2  # Adjust to control the amount of downsampling
HISTORICAL_INTERVAL = "1h"  # Configure the historical data interval, e.g., "1h", "30m", "15m", etc.
START_DATE = "2023-01-01"
END_DATE = "2023-05-15"

# Function to calculate sentiment score using NLTK's Vader SentimentIntensityAnalyzer
def calculate_sentiment_score(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)
    return sentiment_score['compound']

def scrape_news(ticker):
    urls = [
        f"https://finance.yahoo.com/quote/{ticker}",
        f"https://money.cnn.com/quote/quote.html?symb={ticker}",
        # Add more URLs as needed
    ]

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all elements on the page
        elements = soup.find_all()

        titles = [element.text for element in elements]
        descriptions = titles  # Use the same list of titles as descriptions

        if titles:  # If titles were found
            return titles, descriptions

    # If no titles were found at any of the URLs
    raise ValueError(f"No news articles found for {ticker}")

def get_average_sentiment(ticker, company_name):
    # Scrape news articles for the stock ticker and company name
    titles_ticker, descriptions_ticker = scrape_news(ticker)
    titles_company, descriptions_company = scrape_news(company_name)

    if len(titles_ticker) == 0: #and len(titles_company) == 0:
        print(f"No news articles found for {ticker} or {company_name}")
        return None
    else:
        sentiment_scores = []
        for title, description in zip(titles_ticker, descriptions_ticker):
            text = f'{title} {description}'
            sentiment_score = calculate_sentiment_score(text)
            sentiment_scores.append(sentiment_score)
        
        for title, description in zip(titles_company, descriptions_company):
            text = f'{title} {description}'
            sentiment_score = calculate_sentiment_score(text)
            sentiment_scores.append(sentiment_score)
        
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        print(f"Average Sentiment Scores: {average_sentiment}")
        return average_sentiment

def tickers_to_company_names(ticker):
    company_name = []
    stock = yf.Ticker(ticker)
    company_info = stock.info
    if company_info is not None:
        company_name = company_info.get('longName')
    if not company_name:
        print(f"Company name not found for ticker: {ticker}")

    return company_name

# Function to predict stock
def predict_stock(ticker, company_name):
    try:
        # Fetch historical stock data
        df = yf.download(ticker, start=START_DATE, end=END_DATE, interval=HISTORICAL_INTERVAL)

        if df.empty:
            print(f"No historical stock data found for {ticker}")
            return

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

        # Normalize the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df)

        # Add another scaler for 'Close' prices only
        close_scaler = MinMaxScaler()
        df[['Close']] = close_scaler.fit_transform(df[['Close']])
        #df['Close'] = close_scaler.inverse_transform(df[['Close']])

        # Drop the rows with missing values
        #df = df.dropna()

        # Function to create sequences
        def create_sequences(data, sequence_length):
            x = []
            y = []
            for i in range(len(data) - sequence_length - 1):
                x.append(data[i:(i + sequence_length), :])
                y.append(data[i + sequence_length, 0])  # Predict the next 'Close' price
            return np.array(x), np.array(y).reshape(-1, 1)

        # Create sequences
        sequence_length = SEQUENCE_LENGTH  # Adjust sequence length
        x, y = create_sequences(data_scaled, sequence_length)

        # Split the data into training and testing data
        train_length = int(len(df) * 0.7)
        x_train, x_test = x[:train_length], x[train_length:]
        y_train, y_test = y[:train_length], y[train_length:]

        # Build the CNN-GRU model
        model = Sequential()
        for i in range(NUM_CONV_LAYERS):
            model.add(Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu',
                             input_shape=(sequence_length, 8)))
            model.add(MaxPooling1D(pool_size=POOL_SIZE))
        model.add(GRU(LSTM_UNITS, activation='relu', return_sequences=True))
        model.add(Dropout(DROPOUT_RATE))
        model.add(GRU(LSTM_UNITS, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

        # Train the model
        model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

        # Test the model using the test data
        predictions = model.predict(x_test)

        # Reverse the scaling for the predictions
        predictions = close_scaler.inverse_transform(predictions)

        # Now let's use the model to predict the next period
        new_df = data_scaled[-sequence_length:].copy()  # shape: (sequence_length, 8)
        forecast = []

        for _ in range(PERIOD):  # Change this to 7
            new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], new_df.shape[1]))
            predicted_price = model.predict(new_df_scaled)  # shape: (1, 1)
            # Propagate the last features
            last_features = new_df[-1, 1:]
            new_prediction = np.concatenate([predicted_price, last_features.reshape(1, -1)], axis=1)  # shape: (1, 6)
            new_df = np.concatenate([new_df[1:], new_prediction])  # shape: (sequence_length, 6)

            forecast.append(predicted_price[0])

        # Reverse the scaling for the forecast
        forecast = close_scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

        print("The forecast for the next", PERIOD, "days is:", forecast)

        # Create a DataFrame for the last week of actual prices
        last_week = df['Close'].tail(PERIOD)

        # Get the day after the last day in last_week
        start_date = last_week.index[-1] + pd.DateOffset(days=1)

        # Generate the dates for the forecast
        forecast_dates = pd.date_range(start=start_date, periods=PERIOD)

        # Create a DataFrame for the forecast using these dates
        forecast_week = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

        # Concatenate the actual and forecasted prices
        result = pd.concat([last_week, forecast_week], axis=1)

        # Calculate the percentage of change
        result['Percentage Change'] = result['Forecast'].pct_change() * 100
        print(result)

        # Plot the actual, training, testing and forecasted prices
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

    except Exception as e:
        print("An error occurred during stock prediction:", str(e))

# Predict stocks for a specific list of tickers
#prediction_tickers = input("Enter the tickers you want to predict (separated by comma): ").split(',')

for ticker in tickers: #prediction_tickers:
    company_name = tickers_to_company_names(ticker)
    predict_stock(ticker, company_name)
