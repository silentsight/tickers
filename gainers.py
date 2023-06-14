import csv
import requests
from bs4 import BeautifulSoup
import datetime
from urllib.parse import quote


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

# Scrape top gainer stocks
top_gainers = scrape_top_gainers()

current_datetime = datetime.datetime.now()
print(current_datetime.strftime("%Y-%m-%d"))
filename = current_datetime.strftime("%Y%m%d_%H") + "_top_gainers.csv"

# Your list of tickers
tickers = ['WCC','WSFS','CLRO','HRI','GBCI','TGOPY','MRVL','FN', 'UMC','AAPL','BTC']

# Export to CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Stock"])
    writer.writerows([[stock] for stock in top_gainers])
    # Append each ticker as a new row
    for ticker in tickers:
        writer.writerow([ticker])
print("Top gainer stocks exported to", filename)
