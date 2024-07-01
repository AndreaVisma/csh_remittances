"""
Script: data_download.py
Author: Andrea Vismara
Date: 01/07/2024
Description: downloads remittances data from the KNOMAD website (https://www.knomad.org/data/remittances)
"""

import os
import requests

# set the data folder. You might want to check the current working directory
data_folder = os.getcwd() + "\\data\\"

# store the URLs for the three files to download
url_inflows = "https://www.knomad.org/sites/default/files/2024-06/inward-remittance-flows-brief-40-june-2024_1.xlsx"
url_outflows = "https://www.knomad.org/sites/default/files/2024-06/outward-remittance-flows-brief-40-june-2024_0.xlsx"
url_matrix = "https://www.knomad.org/sites/default/files/2022-12/bilateral_remittance_matrix_2021_0.xlsx"

### download inflows
response = requests.get(url_inflows)

if response.ok:
    with open(data_folder + "inward-remittance-flows-2024.xlsx", mode="wb") as file:
        file.write(response.content)
    print(f"Remittances inflow data saved at {data_folder + 'inward-remittance-flows-2024.xlsx'}")
else:
    print("Something went wrong in the remittances inflow data download!")

### download outflows
response = requests.get(url_outflows)

if response.ok:
    with open(data_folder + "outward-remittance-flows-2024.xlsx", mode="wb") as file:
        file.write(response.content)
    print(f"Remittances inflow data saved at {data_folder + 'outward-remittance-flows-2024.xlsx'}")
else:
    print("Something went wrong in the remittances outflow data download!")

### download inflows
response = requests.get(url_matrix)

if response.ok:
    with open(data_folder + "bilateral_remittance_matrix_2021.xlsx", mode="wb") as file:
        file.write(response.content)
    print(f"Remittances inflow data saved at {data_folder + 'bilateral_remittance_matrix_2021.xlsx'}")
else:
    print("Something went wrong in the remittances matrix data download!")