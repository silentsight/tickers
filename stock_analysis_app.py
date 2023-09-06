# Imports
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
import numpy as np
import time
import random
from requests import Session
import csv
import datetime
from urllib.parse import quote
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from threading import Thread
from queue import Queue
from scipy.stats import skew, kurtosis
from ta.momentum import rsi
from ta.trend import macd
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import EMAIndicator

current_datetime = datetime.datetime.now()
HISTORICAL_INTERVAL = "1d"  # Configure the historical data interval, e.g., "1h", "30m", "15m", etc.
START_DATE = "2023-01-01"
END_DATE = current_datetime.strftime("%Y-%m-%d")

def fetch_stock_data(ticker):
    """
    Fetch stock data for a given ticker using Yahoo Finance API.
    """
    stock = yf.download(ticker, start=START_DATE, end=END_DATE, interval=HISTORICAL_INTERVAL)
    return stock

def scrape_top_gainers():
    try:
        url = "https://finance.yahoo.com/gainers"
        encoded_url = quote(url, safe=':/?&=')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(encoded_url, headers=headers)        
        soup = BeautifulSoup(response.text, 'html.parser')
        gainer_elements = soup.find_all(attrs={"data-test": "quoteLink"})
        gainer_stocks = [element.text for element in gainer_elements]

        return gainer_stocks
    except Exception as e:
        print(f"Error occurred while getting top gainers.")
        return []


def fetch_web_sentiment(ticker):
    # Initialize empty string to store all text
    all_text = ""
    
    # BeautifulSoup method for web scraping
    session = Session()
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    
    # Rate limiting
    time.sleep(random.randint(2,5))
    
    # Error handling
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
    
    else:
        # Parsing content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Assuming each post is inside a <p> tag
        posts = soup.find_all("p")
        
        # Aggregate the text from each post into a single string
        extracted_text_bs = " ".join(post.text for post in posts)
        
        # Append to all_text
        all_text += extracted_text_bs
    
    # yfinance method for fetching news
    stock = yf.Ticker(ticker)
    news_data = stock.news
    
    # Aggregate news titles and summaries
    extracted_text_yf = " ".join([(item['title'] + " " + item.get('summary', '')) for item in news_data])
    
    # Append to all_text
    all_text += " " + extracted_text_yf
    
    # print(f"News for {ticker}. News: {all_text}")

    return all_text

def analyze_sentiment(text_data):
     
    # Perform sentiment analysis on the text data gathered from the web using VADER.
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text_data)

    # You can either return the compound score or individual scores for 'neg', 'neu', and 'pos'
    return sentiment_scores['compound']

def analyze_stock(ticker, company_name,timeout_seconds=10):
    # Define a function to be run in a separate thread
    def worker(queue):    
        try:
            # Fetch historical stock data
            df = fetch_stock_data(ticker) #yf.download(ticker, start=START_DATE, end=END_DATE, interval=HISTORICAL_INTERVAL)

            if df.empty:
                print(f"No historical stock data found for {ticker}")
                return
            # Define weights for each factor
            weights = {
                'volatility': 10,  # Lower volatility is generally better for risk-averse investors
                'skewness': 5,  # Extreme skewness can indicate higher risk
                'kurtosis': 5,  # Extreme kurtosis can indicate higher risk
                'recent_performance_1_month': 7,  # Recent performance can be a good indicator, but it's also backward-looking
                'recent_performance_6_month': 8,  # Longer-term performance might be a slightly better indicator
                'trend': 9,  # Following the trend can be a relatively safer strategy
                'momentum': 7,  # Momentum can be important, but it can also change quickly
                'rsi': 7,  # RSI is a useful indicator, but should not be overweighted
                'macd': 7,  # MACD is a useful indicator, but should not be overweighted
                'pe_ratio': 10,  # Financial ratios are important indicators of a company's financial health
                'ps_ratio': 10,  # Financial ratios are important indicators of a company's financial health
                'pb_ratio': 10,  # Financial ratios are important indicators of a company's financial health
                'debt_equity': 10,  # Financial ratios are important indicators of a company's financial health
                'dividend_rate': 8,  # A high dividend rate can be good for risk-averse investors
                'short_interest': 5,  # High short interest can indicate higher risk
                'obv_rate_of_change': 7,  # Volume changes can be indicative, but are often less important than other factors
                'sentiment_score': 10,  # Sentiment can be a strong indicator, especially in the short term
                'ema': 7, # EMA (Exponential Moving Average)
                'boll': 5, # BOLL (Bollinger Bands)
                'vwap': 5 # VWAP (Volume Weighted Average Price)
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

            #average_sentiment = get_average_sentiment(ticker, company_name)
            average_sentiment = analyze_sentiment(fetch_web_sentiment(ticker)) #Compound returned based on VADER NLP

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
            if average_sentiment >= 0.5:
                score += weights['sentiment_score']
            elif average_sentiment < 0.5:
                score -= weights['sentiment_score']  # be careful of the score going negative

            # Normalize score to be between 0 and 100
            score = max(0, min(score, 100))

            # Get recommendation based on score
            recommendation = get_recommendation(score)
            #print("Recommendation:", recommendation)

            # Store final results
            results = pd.DataFrame({
                'Ticker': [ticker],
                'Name': [company_name],
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

            queue.put(results) # Put the result in the queue

        except Exception as e:
            queue.put(e) # Put the exception in the queue if an error occurredprint(f"An error occurred while analyzing {ticker}: {str(e)}")
            print(f"An error occurred while analyzing {ticker}: {str(e)}")
    
    # Create a queue to store the result or exception
    result_queue = Queue()

    # Create a thread to run the worker function
    analysis_thread = Thread(target=worker, args=(result_queue,))

    # Start the thread
    analysis_thread.start()

    # Wait for the thread to finish, with the specified timeout
    analysis_thread.join(timeout=timeout_seconds)

    # Check if the thread is still alive (i.e., it didn't finish in time)
    if analysis_thread.is_alive():
        # If the thread is still running, raise a timeout error
        raise TimeoutError(f"Analysis of {ticker} took too long and was terminated.")

    # Retrieve the result or exception from the queue
    try:
        result_or_exception = result_queue.get(timeout=10)  # 10 seconds timeout
    except result_queue.Empty:
        print("Worker thread did not put any result in the queue within the timeout period.")
        return None


    # If an exception was raised inside the worker function, raise it here
    if isinstance(result_or_exception, Exception):
        raise result_or_exception

    return result_or_exception

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

    """ Now part of analyze stock
def analyze_performance_indicators(stock_data):
    
    # Analyze performance indicators like P/E Ratio, Moving Averages, etc.
    
    # Calculate 20-day moving average
    stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean()
    return {"20_MA": stock_data['20_MA'].iloc[-1]}

def analyze_risk(stock_data):
    
    # Analyze the risk associated with the stock based on historical data.
    
    # Calculate historical volatility as a measure of risk
    stock_data['Log_Ret'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = stock_data['Log_Ret'].std() * np.sqrt(252)  # Annualize
    return {"Volatility": volatility}

def calculate_recommendation_score(performance_metrics, sentiment_score, risk_metrics):
    
    # Calculate a recommendation score based on performance metrics, sentiment, and risk.
    
    # This is a placeholder. You can create a more complex algorithm based on your specific needs.
    score = (performance_metrics['20_MA'] + sentiment_score) / (1 + risk_metrics['Volatility'])
    return score
    """

def tickers_to_company_names(ticker):
    company_name = []
    stock = yf.Ticker(ticker)
    company_info = stock.info
    if company_info is not None:
        company_name = company_info.get('longName')
    if not company_name:
        print(f"Company name not found for ticker: {ticker}")
    return company_name

# Main Function
def main():
    """
    Main function to tie all the steps together.
    """
    # Scrape top gainer stocks
    top_gainers = scrape_top_gainers()

    current_datetime = datetime.datetime.now()
    filename = current_datetime.strftime("%Y%m%d_%H") + "_analysis_results.csv"

    # Your list of extra tickers
    extra_tickers = ['WCC', 'TSM' ,'AMD', 'AMZN', 'NVDA', 'GOOGL', 'AAPL']  # You can add more tickers here

    # Combine top gainers and extra tickers
    all_tickers = list(set(top_gainers + extra_tickers))
    
    # Perform initial analysis for each stock
    all_results = []

    for ticker in all_tickers:
        try:
            company_name = tickers_to_company_names(ticker)
            results_df = analyze_stock(ticker, company_name)
            if results_df is not None:  # Check if DataFrame is not None
                #print("Shape of results_df:", results_df.shape)  # Debug line
                #print("Columns of results_df:", results_df.columns)  # Debug line
                all_results.append(results_df)
            print(f"Completed analysis for {ticker}.")
        except Exception as e:
            print(f"An error occurred while analyzing {ticker}: {str(e)}")

    # Concatenate all DataFrames into a single DataFrame
    if all_results:  # Check if list is not empty
        final_results = pd.concat(all_results, ignore_index=True)
        #print("Shape of final_results:", final_results.shape)  # Debug line
        #print("Columns of final_results:", final_results.columns)  # Debug line

        # Sort and save to CSV
        final_results = final_results.sort_values(by='Score', ascending=False)
        try:
            final_results.to_csv(filename, index=False)
        except Exception as e:
            print(f"An error occurred while printing to csv: {str(e)}")


if __name__ == "__main__":
    main()
