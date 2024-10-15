"""
Script: test the claims of the diaspora model.py
Author: Andrea Vismara
Date: 04/10/2024
Description: test if what we claimed in the diaspora model really explain the last 15 years of migrations
"""

import pandas as pd
import numpy as np
from utils import dict_names
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# historical migrants in Austria
df = pd.read_excel("c:\\data\\migration\\austria\\population_by_nationality_year_land_2010-2024.xlsx")
df = df.melt(id_vars = 'year', value_vars=df.columns[2:],
             value_name='pop', var_name='nationality')
df.loc[df['pop'] == '-', 'pop'] = 0
df = df.groupby(['year', 'nationality']).sum()
df = df.reset_index()
df['nationality'] = df['nationality'].map(dict_names)
df = df.dropna()
df.year = df.year.astype(int) - 1
df.rename(columns = {'nationality':'country'}, inplace = True)
df = df[~df[["country", "year"]].duplicated()]

## estimated daily arrivals (pull factor) in the diaspora model
rho = 3.29 * 1e-4
yearly_rho = ((1 + rho) ** 365) - 1

##estimate forecasts for the whole time series based on the 2009 values
for country in tqdm(df.country.unique()):
    df.loc[(df.country == country) & (df.year == 2009), "est_pop"] =\
        df.loc[(df.country == country) & (df.year == 2009), "pop"]
    yearly_arrivals = df.loc[(df.country == country) & (df.year == 2009), "est_pop"] * yearly_rho
    for year in df.year.unique()[1:]:
        val = (df.loc[(df.country == country) & (df.year == year - 1), "est_pop"].item() +
               yearly_arrivals).item()
        df.loc[(df.country == country) & (df.year == year), "est_pop"] = val

##population growth rate in the data
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, "pop_gr_rate"] = df.loc[df.country == country, "pop"].pct_change()

df = df[df["pop"] != 0]
df = df[df['pop_gr_rate'] != np.inf]
df["pct_error"] = 100 * abs((df["pop"] - df["est_pop"]) / df["pop"])
df = df[df['pct_error'] != np.inf]
df["growth_rate_dev"] = abs(df['pop_gr_rate'] - df['pop_gr_rate'].mean())
df["pct_error"].plot()
plt.show(block = True)

fig, ax = plt.subplots(figsize=(15, 9))
sns.scatterplot(df[df.year > 2009], x="growth_rate_dev", y="pct_error", ax=ax, hue="year")
plt.grid(True)
plt.title(f"Migrant population growth rate vs prediction error of the diaspora model")
plt.xlabel('Population growth rate')
plt.ylabel('Percentage preditcion error')
plt.show(block=True)

df["growth_rate_dev"].plot()
plt.show(block=True)

### yearly totals
df_tot = df[["year", "pop", "est_pop"]].groupby("year").sum().reset_index()
df_tot.rename(columns = {"pop" : "real population", "est_pop" : "estimated population with\nthe diaspora model"},
              inplace = True)
df_tot = pd.melt(df_tot, id_vars="year", value_vars=df_tot.columns[1:], value_name="people", var_name="series")

fig, ax = plt.subplots(figsize=(15, 9))
sns.lineplot(df_tot, x="year", y = "people", hue = "series")
plt.grid(True)
plt.title(f"Migrant population in Austria vs. diaspora model predictions")
plt.xlabel('Year')
plt.ylabel('Number of people')
plt.show(block=True)

def plot_country(country):
    df_country = df[df.country == country].copy()
    df_country.rename(columns={"pop": "real population", "est_pop": "estimated population with\nthe diaspora model"},
                  inplace=True)
    df_melt = pd.melt(df_country, id_vars="year", value_vars=df_country.columns[2:4], value_name="people", var_name="series")

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.lineplot(df_melt, x="year", y="people", hue="series", ax=ax)
    plt.grid(True)
    plt.title(f"Migrant population in Austria vs. diaspora model predictions for the population of {country}")
    plt.xlabel('Year')
    plt.ylabel('Number of people')
    plt.show(block=True)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax2 = ax.twinx()
    sns.lineplot(df_country, x="year", y="pct_error", ax = ax, color = 'red')
    sns.lineplot(df_melt, x="year", y="people", hue="series", ax=ax2)
    plt.grid(True)
    plt.title(f"Percentage error for the diaspora model predictions for {country}")
    plt.xlabel('Year')
    plt.ylabel('Number of people')
    plt.show(block=True)

plot_country("Syria")
plot_country("Ukraine")
plot_country("Italy")
plot_country("Afghanistan")
plot_country("Serbia")
plot_country("Turkey")
plot_country("Germany")
plot_country("Morocco")
plot_country("Romania")