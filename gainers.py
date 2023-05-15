import csv
import requests
from bs4 import BeautifulSoup
import datetime
from urllib.parse import quote


def scrape_top_gainers():
    url = "https://finance.yahoo.com/gainers?count=100"
    encoded_url = quote(url, safe=':/?&=')
    response = requests.get(encoded_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    gainer_elements = soup.find_all(attrs={"data-test": "quoteLink"})
    gainer_stocks = [element.text for element in gainer_elements]

    return gainer_stocks

# Scrape top gainer stocks
top_gainers = scrape_top_gainers()

current_datetime = datetime.datetime.now()

filename = current_datetime.strftime("%Y%m%d_%H") + "_top_gainers.csv"

# Export to CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Stock"])
    writer.writerows([[stock] for stock in top_gainers])

print("Top gainer stocks exported to", filename)
