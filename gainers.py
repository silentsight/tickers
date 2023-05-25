import csv
import requests
from bs4 import BeautifulSoup
import datetime
from urllib.parse import quote


def scrape_top_gainers():
    try:
        url = "https://finance.yahoo.com/gainers"
        encoded_url = quote(url, safe=':/?&=')
        response = requests.get(encoded_url)
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

# Export to CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Stock"])
    writer.writerows([[stock] for stock in top_gainers])

print("Top gainer stocks exported to", filename)
