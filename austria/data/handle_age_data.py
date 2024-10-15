"""
Script: handle_age_data.py
Author: Andrea Vismara
Date: 02/10/2024
Description: clean the age data downloaded from statistik austria
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


raw_folder = "C:\\Data\\population\\austria\\age_raw"
out_folder = "C:\\Data\\population\\austria\\"
list_files = os.listdir(raw_folder)

df_all = pd.DataFrame([])

for file in tqdm(list_files):

    df = pd.read_excel(raw_folder + "\\" + file,
                       skiprows=12, skipfooter=6).iloc[:, 1:]
    df.rename(columns = {"Unnamed: 1" : "year", "Unnamed: 2" : "age_group"},
              inplace = True)

    df = df.iloc[1:]
    df['year'] = df.year.ffill()
    df = df.replace('-', np.nan)

    #sometimes country columns are duplicated
    cols_to_check = [x for x in df.columns.tolist() if '.1' in x]
    for i in cols_to_check:
        if df[i].isna().sum() < df[i.replace('.1', '')].isna().sum():
            col_to_drop = i.replace('.1', '')
            df.drop(columns = col_to_drop, inplace = True)
            df.rename(columns = {i : i.replace('.1', '')}, inplace = True)
        else:
            col_to_drop = i
            df.drop(columns=col_to_drop, inplace=True)

    df.fillna(0, inplace = True)
    df = df.replace('up to 4 years old', '0 to 4 years old')
    df['age_group'] = df['age_group'].apply(lambda x: [int(s) for s in re.findall(r'\b\d+\b', x)])

    df = pd.melt(df, id_vars=df.columns.tolist()[:2], value_vars=df.columns.tolist()[2:],
                 var_name='country', value_name='people')

    df_all = pd.concat([df_all, df])

df_all['mean_age'] = df['age_group'].apply(np.mean)
df_all['year'] = df_all['year'].astype(int) - 1
df_all['country'] = df_all['country'].map(dict_names)
df_all.dropna(inplace = True)

df_all.to_excel(out_folder + "age_nationality_hist.xlsx", index = False)

df = df_all.copy()
def plot_dist_year_country(year, country):
    df_small = df[(df.country == country) & (df.year == year)].copy()
    df_small['ratio'] = 100 * df_small['people'] / df_small.people.sum()

    fig, ax = plt.subplots(figsize = (9,6))
    plt.bar(x = df_small['mean_age'], height = df_small['ratio'], width = 4)
    plt.grid(True)
    plt.title(f"Age distribution of the population of {country} in {year}")
    plt.xlabel('Age')
    plt.ylabel('Share of the diaspora population')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show(block = True)

def plot_comparison_distribution(years, countries):
    df_small = df[(df.country.isin(countries)) & (df.year.isin(years))].copy()
    df_small['ratio'] = 0
    for country in countries:
        for year in years:
            df_small.loc[(df_small.country == country) & (df_small.year == year), 'ratio'] = (
                    100 * df_small.loc[(df_small.country == country) & (df_small.year == year), 'people']
                    / df_small.loc[(df_small.country == country) & (df_small.year == year), 'people'].sum())
    years_str = [str(x) for x in years]

    if len(countries) > 1:
        fig, ax = plt.subplots(figsize = (9,6))
        sns.barplot(df_small, x = 'mean_age', y = 'ratio', hue = 'country', ax = ax)
        plt.grid(True)
        plt.title(f"Comparison of the age distribution of the population\nof {' and '.join(countries)} living in Austria in {' and '.join(years_str)}")
        plt.xlabel('Age')
        plt.ylabel('Share of the diaspora population')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.show(block = True)
    else:
        fig, ax = plt.subplots(figsize = (9,6))
        sns.barplot(df_small, x = 'mean_age', y = 'ratio', hue = 'year', ax = ax)
        plt.grid(True)
        plt.title(f"Comparison of the age distribution of the population\nof {' and '.join(countries)} living in Austria in {' and '.join(years_str)}")
        plt.xlabel('Age')
        plt.ylabel('Share of the diaspora population')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        plt.show(block = True)

plot_comparison_distribution([2020], ['Austria', 'Syria'])