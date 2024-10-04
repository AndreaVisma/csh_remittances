"""
Script: hcpi_gdp_data.py
Author: Andrea Vismara
Date: 03/10/2024
Description: clean the data on the cost of sending remittances by the Wrold Bank
"""

import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from utils import dict_names
import country_converter as coco
cc = coco.CountryConverter()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

folder = "C:\\Data\\remittances\\"
file = "rpw_dataset_2011_2024_q1.xlsx"

xl = pd.ExcelFile(folder + file)
names = xl.sheet_names[-2:]

df_all = pd.DataFrame([])

for name in tqdm(names):
    df = xl.parse(sheet_name=name)
    df_euro = df[df['source_name'].isin(cc.EU28.name_short.to_list()[:-1])].copy()
    df_euro['period'] = df_euro['period'].apply(lambda x: x[:4])
    df_euro = df_euro[['period', 'destination_name', 'cc1 total cost %', 'cc2 total cost %']]
    df_euro.rename(columns={'cc1 total cost %' : 'pct_cost_200usd',
                            'cc2 total cost %' : 'pct_cost_500usd'}, inplace = True)
    df_euro = df_euro.groupby(['period', 'destination_name']).mean().reset_index()

    df_all = pd.concat([df_all, df_euro])

df_all = pd.melt(df_all, id_vars=['period', 'destination_name'], value_vars=['pct_cost_200usd', 'pct_cost_500usd'],
                 var_name="type of transfer", value_name="pct_cost")

df_all.to_excel(folder + "remittances_cost_from_euro.xlsx", index = False)

def plot_country_cost(countries):
    df_country = df_all[df_all.destination_name.isin(countries)].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(df_country, x='period', y='pct_cost', hue='type of transfer', style="destination_name", ax=ax)
    plt.grid(True)
    plt.title(
        f"Cost of sending remittances from the Euro area to \nof {' and '.join(countries)}")
    plt.xlabel('Years')
    plt.ylabel('Percentage of remittance sent')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.show(block=True)