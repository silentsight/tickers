import yfinance as yf
#Grabs latest stock data online

# Define the list of stocks you want to monitor
stocks_to_monitor = ['AAPL', 'MSFT', 'GOOGL']

# Fetch data for each stock
for stock in stocks_to_monitor:
    ticker = yf.Ticker(stock)
    
    # Get historical market data, here max is 1 day
    hist = ticker.history(period="1d")
    today_open = hist['Open'][0]
    today_close = hist['Close'][0]

    # Calculate the price change
    price_change = today_close - today_open

    print(f"Today's price change for {stock}: {price_change}")
