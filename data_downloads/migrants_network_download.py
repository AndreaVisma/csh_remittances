"""
Script: migrants_network_download.py
Author: Andrea Vismara
Date: 04/07/2024
Description: downloads data for the network of migrants globally (https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx)
"""

# TODO: for now it cannot be done, as the UN does not allow the request, so the data needs to be downloaded manually
# import os
# import requests
#
# def download_migration_bilateral_flows():
#     # set the data folder. You might want to check the current working directory
#     data_folder = os.getcwd() + "\\data_downloads\\data\\"
#
#     url_migration_stock = "https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx"
#
#     response = requests.get(url_migration_stock)
#
#     if response.ok:
#         with open(data_folder + "migration_matrix.xlsx", mode="wb") as file:
#             file.write(response.content)
#         print(f"Migration bilateral flows data saved at {data_folder + 'migration_matrix.xlsx'}")
#     else:
#         print("Something went wrong in the bilateral flows data download!")
#
# if __name__ == '__main__':
#     download_migration_bilateral_flows()