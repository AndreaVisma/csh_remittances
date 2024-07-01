"""
Script: migrants_stock_data_download.py
Author: Andrea Vismara
Date: 01/07/2024
Description: runs all the data downloads
"""

from remittances_data_download import download_remittances
from migrants_stock_data_download import download_migration
from gdp_deflator_data_download import download_deflator

def download_all():
    download_remittances()
    download_migration()
    download_deflator()

if __name__ == "__main__":
    download_all()