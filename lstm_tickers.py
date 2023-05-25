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
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import EMAIndicator
from bs4 import BeautifulSoup
import requests
import datetime
from dotenv import load_dotenv
load_dotenv()
import os
MY_GNAPI_KEY = os.environ.get("MY_GNAPI_KEY")

# List of stock tickers you are interested in
#tickers = ['WAL', 'CLRO', 'TGOPY', 'APP', 'AI', 'AMD', 'NVDA', 'RETA', 'MNDY', 'SMCI']
current_datetime = datetime.datetime.now()
# Read the CSV file
tickers_df = pd.read_csv(current_datetime.strftime("%Y%m%d_%H") + "_top_gainers.csv", header=None)

# Convert the DataFrame to a list
tickers = tickers_df[0].tolist()

HISTORICAL_INTERVAL = "1h"  # Configure the historical data interval, e.g., "1h", "30m", "15m", etc.
START_DATE = "2023-01-01"
END_DATE = current_datetime.strftime("%Y-%m-%d")

def fetch_news(ticker):
    api_key = MY_GNAPI_KEY  # Replace with your actual API key
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
        #f"https://old.reddit.com/r/stocks",
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
    try:
        if len(articles) == 0:
            print(f"No news articles found for {ticker} or {company_name}. Changing to alternate.")
            average_sentiment = alt_get_average_sentiment(ticker, company_name)
            return average_sentiment
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
    except Exception as e:
            print(f"Error occurred while getting finantial ratios from Gnews for {ticker}: {str(e)}: Trying alternative.")
            average_sentiment = alt_get_average_sentiment(ticker, company_name)
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
    if score >= 90:
        return "Strong Buy (Very Bullish)"
    elif score >= 80:
        return "Buy (Bullish)"
    elif score >= 70:
        return "Slightly Bullish"
    elif score >= 60:
        return "Neutral (Hold)"
    elif score >= 50:
        return "Slightly Bearish (Hold)"
    elif score >= 40:
        return "Sell (Bearish)"
    else:
        return "Strong Sell (Very Bearish)"


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
        # Define weights for each factor
        weights = {
            'volatility': 5,  # Lower weight as high-risk strategy can tolerate higher volatility
            'skewness': 5,  # Lower weight as high-risk strategy can tolerate higher skewness
            'kurtosis': 5,  # Lower weight as high-risk strategy can tolerate higher kurtosis
            'recent_performance_1_month': 10,  # Higher weight as recent performance can be a strong indicator of future growth
            'recent_performance_6_month': 10,  # Higher weight as longer-term performance can be a strong indicator of future growth
            'trend': 10,  # Higher weight as trend-following can be a successful high-risk strategy
            'momentum': 10,  # Higher weight as momentum is crucial for high-risk strategies
            'rsi': 7,  # Keep the same weight, as RSI is still useful for identifying overbought/oversold conditions
            'macd': 7,  # Keep the same weight, as MACD is still a useful momentum indicator
            'pe_ratio': 5,  # Lower weight as high-risk strategies can tolerate higher P/E ratios
            'ps_ratio': 5,  # Lower weight as high-risk strategies can tolerate higher P/S ratios
            'pb_ratio': 5,  # Lower weight as high-risk strategies can tolerate higher P/B ratios
            'debt_equity': 5,  # Lower weight as high-risk strategies can tolerate higher debt levels
            'dividend_rate': 5,  # Lower weight as high-risk strategies may not prioritize dividends
            'short_interest': 7,  # Keep the same weight, as short interest can still be a useful contrarian indicator
            'obv_rate_of_change': 10,  # Higher weight as volume changes can be indicative of strong future growth
            'sentiment_score': 10,  # Higher weight as sentiment can be a strong indicator, especially for high-risk strategies
            'ema': 10,  # Higher weight as EMA can be a good indicator for trend-following strategies
            'boll': 10,  # Higher weight as Bollinger Bands can be useful for high-risk, high-reward strategies
            'vwap': 10,  # Higher weight as VWAP can be a useful indicator for intraday trading strategies
        }
  
        # Calculate EMA
        ema = EMAIndicator(df['Close'], window=14)
        df['EMA_14'] = ema.ema_indicator()

        # Calculate Bollinger Bands
        bollinger = BollingerBands(df['Close'], window=20)
        df['BOLL_MID'] = bollinger.bollinger_mavg()
        df['BOLL_UPPER'] = bollinger.bollinger_hband()
        df['BOLL_LOWER'] = bollinger.bollinger_lband()

        # Calculate VWAP
        vwap = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'])
        df['VWAP'] = vwap.volume_weighted_average_price()
        
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

        # Add On-Balance Volume (OBV) for volume analysis
        obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
        df['OBV'] = obv.on_balance_volume()

        # Calculate the 5-day rate of change in OBV
        df['OBV Rate of Change'] = df['OBV'].pct_change(periods=5)

        # Fetch additional financial data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Add beta
        df['Beta'] = info.get('beta', np.nan)

        # Fetch financial ratios and other financial data
        try:
            pe_ratio = info.get('trailingPE', np.nan)
            ps_ratio = info.get('priceToSalesTrailing12Months', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            debt_equity_ratio = info.get('debtToEquity', np.nan)
            dividend_rate = info.get('dividendRate', np.nan)
        except Exception as e:
                print(f"An error occurred while getting finantial ratios for {ticker}: {str(e)}")

        # Fetch short interest ratio
        short_interest_ratio = info.get('shortPercentOfFloat', np.nan)

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
        # Add new thresholds for financial ratios and short interest
        pe_ratio_threshold = 14  # example threshold
        ps_ratio_threshold = 3  # example threshold
        pb_ratio_threshold = 3  # example threshold
        debt_equity_ratio_threshold = 1  # example threshold
        dividend_rate_threshold = 0.02  # example threshold
        short_interest_ratio_threshold = 0.1  # example threshold
        obv_rate_of_change_threshold = 0.05  # adjust as needed
        # Add new thresholds for EMA, BOLL, and VWAP
        ema_threshold = 1.05  # Latest closing price is at least 5% above EMA threshold value
        boll_threshold = 1.05  # Latest closing price is at least 5% above BOLL threshold value
        vwap_threshold = 1.05  # Latest closing price is at least 5% aboveVWAP threshold value

        # Initialize score
        score = 0

        recent_performance_1_month = df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1
        recent_performance_6_month = df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1
        momentum = df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1
        # Calculate latest adjusted price
        latest_adjusted_price = df['Close'].iloc[-1]

        average_sentiment = get_average_sentiment(ticker, company_name) #get_average_sentiment(ticker, company_name)

        # Calculate score for each factor
        if df['Volatility'].iloc[-1] < volatility_threshold:
            score += weights['volatility']
        if skewness > skewness_threshold:
            score += weights['skewness']
        if kurt < kurtosis_threshold:
            score += weights['kurtosis']
        if recent_performance_1_month > recent_performance_threshold_1_month:
            score += weights['recent_performance_1_month']
        if recent_performance_6_month > recent_performance_threshold_6_month:
            score += weights['recent_performance_6_month']
        if df['Close'].iloc[-1] > trend_threshold * df['MA_50'].iloc[-1]:
            score += weights['trend']
        if momentum > momentum_threshold:
            score += weights['momentum']
        if df['RSI'].iloc[-1] > rsi_threshold:
            score += weights['rsi']
        if df['MACD'].iloc[-1] > macd_threshold:
            score += weights['macd']
        if pe_ratio < pe_ratio_threshold:
            score += weights['pe_ratio']
        if ps_ratio < ps_ratio_threshold:
            score += weights['ps_ratio']
        if pb_ratio < pb_ratio_threshold:
            score += weights['pb_ratio']
        if debt_equity_ratio < debt_equity_ratio_threshold:
            score += weights['debt_equity']
        if dividend_rate > dividend_rate_threshold:
            score += weights['dividend_rate']
        if short_interest_ratio < short_interest_ratio_threshold:
            score += weights['short_interest']
        if df['OBV Rate of Change'].iloc[-1] > obv_rate_of_change_threshold:
            score += weights['obv_rate_of_change']

        if df['EMA_14'].iloc[-1] > ema_threshold * df['Close'].iloc[-1]:
            score += weights['ema']
        if df['BOLL_MID'].iloc[-1] > boll_threshold * df['Close'].iloc[-1]:
            score += weights['boll']
        if df['VWAP'].iloc[-1] > vwap_threshold * df['Close'].iloc[-1]:
            score += weights['vwap']

        # Adjust the score based on sentiment analysis
        if average_sentiment >= 0.1:
            score += weights['sentiment_score']
        elif average_sentiment < 0.01:
            score -= weights['sentiment_score']  # be careful of the score going negative

        # Normalize score to be between 0 and 100
        score = max(0, min(score, 100))

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
            'OBV': [df['OBV'].iloc[-1]],
            'Beta': [df['Beta'].iloc[-1]],
            'P/E': [pe_ratio],
            'P/S': [ps_ratio],
            'P/B': [pb_ratio],
            'Debt/Eq': [debt_equity_ratio],
            'Dividend rate': [dividend_rate],
            'Short Interest Ratio': [short_interest_ratio],
            'EMA': [df['EMA_14'].iloc[-1]],
            'BOLL': [df['BOLL_MID'].iloc[-1]],
            'VWAP': [df['VWAP'].iloc[-1]],
            'Score': [score],
            'Sentiment Score': [average_sentiment],
            'Recommendation': [recommendation]  # Add this line
        })
        return results

        plot_yn = 0
        if plot_yn == "y":
            # Plot the stock analysis
            plot_stock_analysis(ticker, df)

    except Exception as e:
        print(f"An error occurred while analyzing {ticker}: {str(e)}")

    #return score

# Perform initial analysis for each stock
all_results = []  # List to store the results DataFrames

for ticker in tickers:
    try:
        company_name = tickers_to_company_names(ticker)
        results_df = analyze_stock(ticker, company_name)
        all_results.append(results_df)  # Append the result DataFrame to the list
    except Exception as e:
        print(f"An error occurred while analyzing {ticker}: {str(e)}")

# Concatenate all results DataFrames
final_results = pd.concat(all_results)

# Sort the DataFrame based on the score and display the result
final_results = final_results.sort_values(by='Score', ascending=False)
print(final_results)

current_datetime = datetime.datetime.now()

filename = current_datetime.strftime("%Y%m%d_%H") + "_risky_analysis_results.csv"
# Export the DataFrame to a CSV file
final_results.to_csv(filename, index=False)
