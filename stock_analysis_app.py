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

def fetch_stock_data(ticker):
    """
    Fetch stock data for a given ticker using Yahoo Finance API.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

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


def analyze_performance_indicators(stock_data):
    """
    Analyze performance indicators like P/E Ratio, Moving Averages, etc.
    """
    # Calculate 20-day moving average
    stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean()
    return {"20_MA": stock_data['20_MA'].iloc[-1]}


def analyze_sentiment(text_data):
    """
    Perform sentiment analysis on the text data gathered from the web using VADER.
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text_data)

    # You can either return the compound score or individual scores for 'neg', 'neu', and 'pos'
    return sentiment_scores['compound']

def analyze_risk(stock_data):
    """
    Analyze the risk associated with the stock based on historical data.
    """
    # Calculate historical volatility as a measure of risk
    stock_data['Log_Ret'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = stock_data['Log_Ret'].std() * np.sqrt(252)  # Annualize
    return {"Volatility": volatility}

def calculate_recommendation_score(performance_metrics, sentiment_score, risk_metrics):
    """
    Calculate a recommendation score based on performance metrics, sentiment, and risk.
    """
    # This is a placeholder. You can create a more complex algorithm based on your specific needs.
    score = (performance_metrics['20_MA'] + sentiment_score) / (1 + risk_metrics['Volatility'])
    return score

# Export to Spreadsheet function modified to accept filename
def export_to_spreadsheet(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

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
    extra_tickers = ['AAPL', 'GOOGL', 'AMZN']  # You can add more tickers here

    # Combine top gainers and extra tickers
    all_tickers = list(set(top_gainers + extra_tickers))
    results = []
    
    for ticker in all_tickers:
        print(f"Analyzing {ticker}...")
        
        # Step 1: Data Gathering
        stock_data = fetch_stock_data(ticker)
        web_data = fetch_web_sentiment(ticker)

        # Step 2: Data Analysis
        performance_metrics = analyze_performance_indicators(stock_data)
        sentiment_score = analyze_sentiment(web_data)
        risk_metrics = analyze_risk(stock_data)

        # Step 3: Scoring Algorithm
        recommendation_score = calculate_recommendation_score(performance_metrics, sentiment_score, risk_metrics)
        
        # Append to results
        results.append({
            "Ticker": ticker,
            "20_MA": performance_metrics['20_MA'],
            "Sentiment_Score": sentiment_score,
            "Volatility": risk_metrics['Volatility'],
            "Recommendation_Score": recommendation_score
        })
        
        print(f"Completed analysis for {ticker}.")
    
    # Step 4: Export to Spreadsheet
    export_to_spreadsheet(results, filename)


if __name__ == "__main__":
    main()
