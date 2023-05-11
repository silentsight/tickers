import yfinance as yf
import pandas as pd

# Define the list of stocks you want to monitor
stocks_to_monitor = ['INTA', 'NTCO', 'DVA', 'AKRO', 'CLRO']

# Create a list to store the stock data
data = []

for stock in stocks_to_monitor:
    try:
        ticker = yf.Ticker(stock)
        
        # Get historical market data, here max is 1 day
        hist = ticker.history(period="1d")
        
        if len(hist) == 0:
            print(f"No data was found for {stock} today.")
            continue

        today_open = hist['Open'][0]
        today_close = hist['Close'][0]
        today_high = hist['High'][0]
        today_low = hist['Low'][0]
        
        # Check if 'regularMarketPrice' is available
        if 'regularMarketPrice' in ticker.info:
            current_price = ticker.info['regularMarketPrice']
        else:
            current_price = today_close
        
        # Calculate the price change
        price_change = current_price - today_open

        # Get the news, analysis, and recommendations data
        news = ticker.get_news()
        #analysis = ticker.
        #recommendations = ticker.get_recommendations()

        # Append the data to the list
        data.append({
            'Stock': stock,
            'Open': today_open,
            'High': today_high,
            'Low': today_low,
            'Close': today_close,
            'Current': current_price,
            'Change': price_change,
            #'Analysis': analysis,
            #'Recommendations': recommendations
        })

         # Print the news in a readable format
         
        # Print the news in a readable format
        print(f"News for {stock}:")
        for news_item in news:
            print(f"Title: {news_item['title']}")
            print(f"Link: {news_item['link']}")
            print(f"Published At: {news_item['providerPublishTime']}")
            print()
        
    except Exception as e:
        print(f"An error occurred while fetching data for {stock}: {e}")

        
    except Exception as e:
        print(f"An error occurred while fetching data for {stock}: {e}")

# Create a DataFrame from the list
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
