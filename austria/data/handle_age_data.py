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
from utils import get_quarter, get_last_day_of_the_quarter
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

## interpolate quarterly age info
df_age = pd.read_excel("C:\\Data\\population\\austria\\age_nationality_hist.xlsx")
dependent = ['[0, 4]', '[5, 9]', '[10, 14]', '[65, 69]', '[70, 74]', '[75, 79]',
             '[80, 84]', '[85, 89]', '[90, 94]', '[95, 99]', '[100]']
df_age = df_age[df_age.age_group.isin(dependent)][["year", "country", "people"]].groupby(["year", "country"]).sum().reset_index().merge(
    df_age[~df_age.age_group.isin(dependent)][["year", "country", "people"]].groupby(["year", "country"]).sum().reset_index(),
    on = ["year", "country"]
)
df_age["dep_ratio"] = 100 * df_age.people_x / (df_age.people_x + df_age.people_y)
df_age = df_age[["year", "country", "dep_ratio"]].sort_values(["country", "year"])
df_age.year = pd.to_datetime(df_age.year, format="%Y").map(get_last_day_of_the_quarter)

# Prepare DataFrame for quarterly interpolation
df_q = pd.DataFrame()
for country in tqdm(df_age['country'].unique()):
    df_country = df_age[df_age['country'] == country].copy()
    df_country.set_index(["year"], inplace=True)
    df_country = df_country.asfreq('Q')
    df_country['dep_ratio'] = df_country['dep_ratio'].resample("m").interpolate(method="time")
    df_country['country'] = country
    df_q = pd.concat([df_q, df_country])

df_q = df_q.reset_index()
df_q['quarter'] = df_q.year.map(get_quarter)
df_q['year'] = df_q.year.apply(lambda x: x.year)

df_q.to_excel("c:\\data\\population\\austria\\age_nationality_hist_quarterly.xlsx")

###
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