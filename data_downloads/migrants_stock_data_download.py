"""
Script: migrants_stock_data_download.py
Author: Andrea Vismara
Date: 01/07/2024
Description: downloads data on the stock of migrants by country from the World Bank (https://data.worldbank.org/indicator/SM.POP.TOTL)
"""

import os
import requests

# set the data folder. You might want to check the current working directory
data_folder = os.getcwd() + "\\data_downloads\\data\\"

url_migration_stock = "https://api.worldbank.org/v2/en/indicator/SM.POP.TOTL?downloadformat=excel"

response = requests.get(url_migration_stock)

if response.ok:
    with open(data_folder + "migration_stock_abs.xls", mode="wb") as file:
        file.write(response.content)
    print(f"Migration stock data saved at {data_folder + 'migration_stock_abs.xls'}")
else:
    print("Something went wrong in the migration stock data download!")