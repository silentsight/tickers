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

def fetch_news(ticker):
    api_key = 'cb97ada7f81ce1322db4127be756fa8d'  # Replace with your actual API key
    url = f'https://gnews.io/api/v4/search?q={ticker}&token={api_key}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

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

def alt_get_average_sentiment(ticker, company_name):
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


def get_average_sentiment(ticker, company_name):
    # Fetch news articles for the stock ticker and company name
    articles = fetch_news(ticker) + fetch_news(company_name)

    if len(articles) == 0:
        print(f"No news articles found for {ticker} or {company_name}")
        return None
    else:
        sentiment_scores = []
        for article in articles:
            title = article['title']
            description = article['description']
            content = article['content']
            text = f'{title} {description} {content}'
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

def get_recommendation(score):
    if score >= 5:
        return "Strong Buy (Bullish)"
    elif score >= 3:
        return "Buy (Bullish)"
    elif score <= -5:
        return "Strong Sell (Bearish)"
    elif score <= -3:
        return "Sell (Bearish)"
    else:
        return "Hold"

def plot_stock_analysis(ticker, df):
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
    plt.show()

def analyze_stock(ticker, company_name):
    try:
        # Fetch historical stock data
        df = yf.download(ticker, start=START_DATE, end=END_DATE, interval=HISTORICAL_INTERVAL)

        if df.empty:
            print(f"No historical stock data found for {ticker}")
            return

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

        # Calculate skewness and kurtosis
        skewness = skew(df['Log_Returns'].dropna())
        kurt = kurtosis(df['Log_Returns'].dropna())

        # Recommendation score based on volatility, skewness, kurtosis, recent performance, trend, and momentum
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

        momentum = df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1
        if momentum > momentum_threshold:
            score += 1

        # Check RSI
        if df['RSI'].iloc[-1] > rsi_threshold:
            score += 1

        # Check MACD
        if df['MACD'].iloc[-1] > macd_threshold:
            score += 1
        
        # Calculate latest adjusted price
        latest_adjusted_price = df['Close'].iloc[-1]

        average_sentiment = alt_get_average_sentiment(ticker, company_name) #get_average_sentiment(ticker, company_name)

        # Adjust the score based on sentiment analysis
        if average_sentiment >= 0.5:
            score += 1
        elif average_sentiment < 0:
            score -= 1

        # Get recommendation based on score
        recommendation = get_recommendation(score)
        #print("Recommendation:", recommendation)

        # Store final results
        results = pd.DataFrame({
            'Ticker': [ticker],
            'Latest Adjusted Price': [latest_adjusted_price],
            'Volatility': [df['Volatility'].iloc[-1]],
            'Skewness': [skewness],
            'Kurtosis': [kurt],
            '1-Month Performance': [recent_performance_1_month],
            '6-Month Performance': [recent_performance_6_month],
            'Trend': [df['Close'].iloc[-1] / df['MA_50'].iloc[-1] - 1],
            'Momentum': [momentum],
            'RSI': [df['RSI'].iloc[-1]],
            'MACD': [df['MACD'].iloc[-1]],
            'Score': [score],
            'Sentiment Score': [average_sentiment],
            'Recommendation': [recommendation]  # Add this line
        })
        print(results)

        # Get recommendation based on score
        recommendation = get_recommendation(score)
        print("Recommendation:", recommendation)

        plot_yn = 0 #input("Do you want to plot? y/n: ")
        if plot_yn == "y":
            # Plot the stock analysis
            plot_stock_analysis(ticker, df)

        return score

    except Exception as e:
        print(f"An error occurred while analyzing {ticker}: {str(e)}")

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

        for _ in range(PERIOD):
            new_df_scaled = np.reshape(new_df, (1, new_df.shape[0], new_df.shape[1]))
            predicted_price = model.predict(new_df_scaled)  # shape: (1, 1)
            # Propagate the last features
            last_features = new_df[-1, 1:]
            new_prediction = np.concatenate([predicted_price, last_features.reshape(1, -1)], axis=1)
            new_df = np.concatenate([new_df[1:], new_prediction])  # shape: (sequence_length, 8)

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

        # Plot the actual, training, testing, and forecasted prices
        plt.figure(figsize=(12, 8))
        plt.plot(df.index[sequence_length:sequence_length + len(y_train)],
                 close_scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Training Data')
        plt.plot(df.index[sequence_length + len(y_train) + 1:],
                 close_scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Testing Data')
        plt.plot(df.index[sequence_length + len(y_train) + 1:], predictions, color='red', label='Predicted Price')
        plt.plot(forecast_week.index, forecast_week['Forecast'], color='orange', label='Forecasted Price')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    except Exception as e:
        print("An error occurred during stock prediction:", str(e))



# Perform initial analysis for each stock
"""
for ticker in tickers:
    company_name = tickers_to_company_names(ticker)
    analyze_stock(ticker, company_name)
    print()
"""
# Predict stocks for a specific list of tickers
#prediction_tickers = input("Enter the tickers you want to predict (separated by comma): ").split(',')

for ticker in tickers: #prediction_tickers:
    company_name = tickers_to_company_names(ticker)
    predict_stock(ticker, company_name)
