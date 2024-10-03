"""
Script: handle_age_data.py
Author: Andrea Vismara
Date: 03/10/2024
Description: clean the sex data downloaded from statistik austria
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

folder = "C:\\Data\\population\\austria\\"
file = "sex_raw\\gender_composition_nationality_hist.xlsx"

df = pd.read_excel(folder + file,
                   skiprows=10, skipfooter=9).iloc[:, 1:]
df.rename(columns={"Unnamed: 1": "year", "Unnamed: 2": "sex"},
          inplace=True)
df = df.iloc[1:]

df['year'] = df.year.ffill()
df = df.replace('-', np.nan)

# sometimes country columns are duplicated
cols_to_check = [x for x in df.columns.tolist() if '.1' in x]
for i in cols_to_check:
    if df[i].isna().sum() < df[i.replace('.1', '')].isna().sum():
        col_to_drop = i.replace('.1', '')
        df.drop(columns=col_to_drop, inplace=True)
        df.rename(columns={i: i.replace('.1', '')}, inplace=True)
    else:
        col_to_drop = i
        df.drop(columns=col_to_drop, inplace=True)

df.fillna(0, inplace = True)

df = pd.melt(df, id_vars=df.columns.tolist()[:2], value_vars=df.columns.tolist()[2:],
             var_name='country', value_name='people')
df = df.sort_values(['country', 'year'])
df['year'] = df['year'].astype(int)

df['country'] = df['country'].map(dict_names)
df.dropna(inplace = True)

df.to_excel(folder + "sex_nationality_hist.xlsx", index = False)

def plot_sex_over_time(country):

    df_country = df[df.country == country].copy()

    for year in df_country.year.unique():
        df_country.loc[(df_country.country == country) & (df_country.year == year), 'ratio'] = (
                100 * df_country.loc[(df_country.country == country) & (df_country.year == year), 'people']
                / df_country.loc[(df_country.country == country) & (df_country.year == year), 'people'].sum())

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(df_country, x='year', y='people', hue='sex', ax=ax)
    plt.grid(True)
    plt.title(
        f"Comparison of the sex distribution of the diaspora population of {country} in Austria")
    plt.xlabel('Years')
    plt.ylabel('Number of people')
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.show(block=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(df_country, x='year', y='ratio', hue='sex', ax=ax)
    plt.grid(True)
    plt.title(
        f"Comparison of the sex distribution of the diaspora population of {country} in Austria")
    plt.xlabel('Years')
    plt.ylabel('Percentage of the diaspora population')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.show(block=True)