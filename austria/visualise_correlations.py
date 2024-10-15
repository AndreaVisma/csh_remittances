"""
Script: visualise_correlations.py
Author: Andrea Vismara
Date: 04/10/2024
Description: visualise correlations between remittances data and other variables
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
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from array import array
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset


##remittances and population data
df = pd.read_excel("c:\\data\\remittances\\austria\\remittances_migrant_pop_austria_2011-2023.xlsx")
df = df[df["Remittances flow"] == "from Austria"].drop(columns = "Remittances flow")

##cost data
df_cost = pd.read_excel("C:\\Data\\remittances\\remittances_cost_from_euro.xlsx")
df_cost.rename(columns = {"destination_name" : "country", "period" : "year"}, inplace = True)
df_cost = df_cost[['year', 'country', 'pct_cost']].groupby(['year', 'country']).mean().reset_index()

df = df.merge(df_cost, on= ["country", "year"], how = "left")

##inflation in origin countries
df_inf = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx")
df_inf.rename(columns = {"Country" : "country"}, inplace = True)

df = df.merge(df_inf, on= ["country", "year"], how = "left")

##gdp in origin country

##remittances_per_migrant
df["rem_per_migrant"] = df["mln_euros"] * 1_000_000 / df["pop"]

fig, ax = plt.subplots(figsize=(15, 9))
sns.regplot(df[(~df.hcpi.isna()) & (~df.rem_per_migrant.isna())], x='hcpi', y='rem_per_migrant', ax = ax, fit_reg = True)
plt.grid(True)
plt.title(f"hcpi v remittances sent")
plt.xlabel('HCPI index')
plt.ylabel('remittance sent per migrant')
plt.show(block=True)

sns.lmplot(df[(~df.hcpi.isna()) & (~df.rem_per_migrant.isna())], x='hcpi', y='rem_per_migrant', hue = 'country')
plt.grid(True)
plt.title(f"hcpi v remittances sent")
plt.xlabel('HCPI index')
plt.ylabel('remittance sent per migrant')
plt.show(block=True)


#####
# cluster population evolution
#####
df_pop = pd.pivot_table(df, index="year", columns="country", values="pop").T
array_pop = df_pop.to_numpy()
time_series_pop = to_time_series_dataset(array_pop)
scaler = TimeSeriesScalerMeanVariance()
time_series_dataset_scaled = scaler.fit_transform(time_series_pop)
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
labels = model.fit_predict(time_series_dataset_scaled)

dict_country_label = {}
for i in range(len(df_pop.T.columns)):
    dict_country_label[df_pop.T.columns[i]] = labels[i]

df["pop_label"] = df["country"].map(dict_country_label)
df["pop_label"] = df["pop_label"].astype(str)

#diaspora population
fig = px.scatter(df, x="pop", y="mln_euros", color = 'pop_label')
fig.show()

sns.lmplot(df_norm, x='pop', y='mln_euros', hue = 'country')
plt.grid(True)
plt.title(f"population v remittances sent")
plt.xlabel('diaspora population')
plt.ylabel('remittance sent')
plt.legend('',frameon=False)
plt.show(block=True)

#####
# visualise relations
#####

#diaspora population
fig, ax = plt.subplots(figsize=(15, 9))
sns.regplot(df, x='pop', y='mln_euros', ax = ax)
plt.grid(True)
plt.title(f"population v remittances sent")
plt.xlabel('diaspora population')
plt.ylabel('remittance sent')
plt.show(block=True)

sns.lmplot(df, x='pop', y='mln_euros', hue = 'country')
plt.grid(True)
plt.title(f"population v remittances sent")
plt.xlabel('diaspora population')
plt.ylabel('remittance sent')
plt.legend('',frameon=False)
plt.show(block=True)

#cost of remitting
fig, ax = plt.subplots(figsize=(15, 9))
sns.regplot(df[df['mln_euros'] < 25], x='pct_cost', y='mln_euros', ax = ax)
plt.grid(True)
plt.title(f"cost of remitting v remittances sent")
plt.xlabel('cost of remitting (%)')
plt.ylabel('remittance sent')
plt.show(block=True)

sns.lmplot(df[df['mln_euros'] < 25], x='pct_cost', y='mln_euros', hue = 'country')
plt.grid(True)
plt.title(f"cost of remitting v remittances sent")
plt.xlabel('cost of remitting (%)')
plt.ylabel('remittance sent')
plt.legend('',frameon=False)
plt.show(block=True)