"""
Script: gdp_deflator_data_download.py
Author: Andrea Vismara
Date: 01/07/2024
Description: downloads GDP deflator data from the World Bank's website (https://data.worldbank.org/indicator/NY.GDP.DEFL.ZS?locations=US)
"""

import os
import requests

# set the data folder. You might want to check the current working directory
data_folder = os.getcwd() + "\\data_downloads\\data\\"

deflator_url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.DEFL.ZS?downloadformat=excel"

response = requests.get(deflator_url)

if response.ok:
    with open(data_folder + "gdp_deflator.xls", mode="wb") as file:
        file.write(response.content)
    print(f"GDP deflator data saved at {data_folder + 'gdp_deflator.xls'}")
else:
    print("Something went wrong in the GDP deflator data download!")