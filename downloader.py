import yfinance as yf
import pandas as pd

def download_data(ticker):
    # Download historical data as a DataFrame
    df = yf.download(ticker,'2015-01-01','2023-04-30')
    
    # Select relevant columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save to a CSV file
    df.to_csv(f"{ticker}.csv")

# List of stock tickers you are interested in
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'DVA']

# Download data for each stock
for ticker in tickers:
    download_data(ticker)