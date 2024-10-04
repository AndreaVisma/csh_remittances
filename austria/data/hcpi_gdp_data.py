"""
Script: hcpi_gdp_data.py
Author: Andrea Vismara
Date: 03/10/2024
Description: clean the HCPI data from He et al. (2023)
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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

folder = "C:\\Data\\economic\\"
#inflation
file = "global_inflation_data.xlsx"

df = pd.read_excel(folder + file, sheet_name= 'hcpi_a', skipfooter=2, usecols="C:BG")
df.drop(columns = df.columns.tolist()[1:3], inplace = True)

df = pd.melt(df, id_vars='Country', value_vars=df.columns.tolist()[1:],
             var_name='year', value_name='hcpi')
df.sort_values(['Country', 'year'], inplace = True)

df['Country'] = df["Country"].map(dict_names)
df.dropna(inplace = True)

df.to_excel(folder + "annual_inflation_clean.xlsx", index = False)

#gdp
file = "global_gdp.xls"

df = pd.read_excel(folder + file, sheet_name= 'Data', skiprows=3)
df.drop(columns = ["Country Code", "Indicator Name", "Indicator Code"], inplace = True)

df = pd.melt(df, id_vars='Country Name', value_vars=df.columns.tolist()[1:],
             var_name='year', value_name='gdp')
df.rename(columns = {"Country Name" : "country"}, inplace = True)
df.sort_values(['country', 'year'], inplace = True)

df['country'] = df["country"].map(dict_names)
df["year"] = df["year"].astype(int)
df = df[df.year >= 2000]
df = df[~df.country.isna()]

df.to_excel(folder + "annual_gdp_clean.xlsx", index = False)

def plot_country_inflation(country):

    fig, ax = plt.subplots(figsize=(9, 6))
    if isinstance(country, list):
        df_country = df[df.Country.isin(country)].copy()
        sns.lineplot(df_country, x='year', y='hcpi', hue='Country', ax=ax)
        plt.title(
            f"HCPI index for {', '.join(country)}")
    else:
        df_country = df[df.Country == country].copy()
        sns.lineplot(df_country, x='year', y='hcpi', ax=ax)
        plt.title(
            f"HCPI index for {country}")
    plt.grid(True)
    plt.xlabel('Years')
    plt.ylabel('Yearly inflation')
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show(block=True)
