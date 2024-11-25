"""
Script: merge_panel_data.py
Author: Andrea Vismara
Date: 04/10/2024
Description: merge all the data into one panel
"""

import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from utils import dict_names, austria_nighbours, find_outliers_iqr
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
df.describe()

##cost data
df_cost = pd.read_excel("C:\\Data\\remittances\\remittances_cost_from_euro.xlsx")
df_cost.rename(columns = {"destination_name" : "country", "period" : "year"}, inplace = True)
df_cost = df_cost[['year', 'country', 'pct_cost']].groupby(['year', 'country']).mean().reset_index()
df_cost.describe()

df = df.merge(df_cost, on= ["country", "year"], how = "left")

##inflation in origin countries
df_inf = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx")
df_inf.rename(columns = {"Country" : "country"}, inplace = True)
df_inf.describe()
df_inf['hcpi_cap'] = df_inf['hcpi']
df_inf.loc[df_inf['hcpi_cap'] > 500, 'hcpi_cap'] = 500
find_outliers_iqr(df_inf['hcpi'])
fig = px.box(df_inf, y='hcpi_cap')
# fig.show()
df = df.merge(df_inf, on= ["country", "year"], how = "left")

##gdp in origin country
df_gdp = pd.read_excel("C:\\Data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx")
df_gdp = df_gdp.ffill()
fig = px.box(df_gdp, y='gdp')
# fig.show()
df = df.merge(df_gdp, on= ["country", "year"], how = "left")

## gender composition


## age data
df_age = pd.read_excel("C:\\Data\\population\\austria\\age_nationality_hist.xlsx")
dependent = ['[0, 4]', '[5, 9]', '[10, 14]', '[65, 69]', '[70, 74]', '[75, 79]',
             '[80, 84]', '[85, 89]', '[90, 94]', '[95, 99]', '[100]']
df_age = df_age[df_age.age_group.isin(dependent)][["year", "country", "people"]].groupby(["year", "country"]).sum().reset_index().merge(
    df_age[~df_age.age_group.isin(dependent)][["year", "country", "people"]].groupby(["year", "country"]).sum().reset_index(),
    on = ["year", "country"]
)
df_age["dep_ratio"] = 100 * df_age.people_x / (df_age.people_x + df_age.people_y)
df_age = df_age[["year", "country", "dep_ratio"]].sort_values(["country", "year"])
df = df.merge(df_age, on= ["country", "year"], how = "left")

## dummy for neighbouring countries
df["neighbour_dummy"] = np.where(df["country"].isin(austria_nighbours), 1, 0)

## simulated incomes
incomes_list = pd.read_excel("c:\\data\\economic\\austria\\avg_annual_hh_income.xlsx")["income"].tolist()
df_class = pd.read_excel("c:\\data\\economic\\income_classification_countries_wb.xlsx", usecols="A:B", skipfooter=49)
df_class['country'] = df_class['country'].map(dict_names)
df = df.merge(df_class, on = 'country', how = 'left')
dict_income = {'High income' : 1, 'Upper middle income' : 0.85, 'Lower middle income' : 0.75, 'Low income' : 0.65}
df_class['mult'] = df_class['group'].map(dict_income)
years = df.year.unique().tolist()
df_inc = pd.DataFrame([])
for country in tqdm(df.country.unique()):
    mult_country = df_class[df_class.country == country]["mult"].item()
    incomes = [np.random.poisson(incomes_list[i] * mult_country) for i in range(len(incomes_list))]
    df_country = pd.DataFrame({"year" : years, "income" : incomes})
    df_country["country"] = country
    df_inc = pd.concat([df_inc, df_country])

def plot_income_country_v_austria(country):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(years, incomes_list, label = "average income Austria")
    ax.plot(years, df_inc[df_inc.country == country]["income"], label=f"average income migrants from {country}")
    plt.grid(True)
    plt.xlabel('Years')
    plt.ylabel('Euros')
    plt.legend()
    plt.show(block=True)
# plot_income_country_v_austria("Bosnia")

df = df.merge(df_inc, on= ["country", "year"], how = "left")

##natural disasters
df_nd = pd.read_excel("C:\\Data\\natural_disasters\\emdat_country_type_quarterly.xlsx")
#clean dates
df_nd[['Start Year','Start Month', 'Start Day']] = (
    df_nd[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
df_nd.rename(columns = dict(zip(['Start Year','Start Month', 'Start Day'],
                              ["year", "month", "day"])), inplace = True)
df_nd["date_start"] = pd.to_datetime(df_nd[["year", "month", "day"]])
df_nd.drop(columns = ["year", "month", "day", "quarter"], inplace = True)
df_nd['year'] = df_nd['date_start'].dt.year
df_nd.rename(columns = {'Country' : 'country'}, inplace = True)
## country population
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
#merge
df_nd = df_nd.merge(df_pop_country, on=['country', 'year'], how = 'left')

#percentage affected dataframe
df_nd_pct = df_nd.copy()
cols = ['Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected']
for col in cols:
    df_nd_pct[col] = 100 * df_nd_pct[col] / df_nd_pct['population']
df_nd_pct.dropna(inplace = True)
cols = ['country', 'Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected', 'year']
df_group = df_nd_pct[cols].groupby(['country', 'year']).sum().reset_index()
df = df.merge(df_group, on = ['country', 'year'], how = 'left')
df.fillna(0, inplace = True)

## clean and save
# give to all countries in a certain income group the same cost
for group in tqdm(df_class['group'].unique()):
    group_countries = df_class[df_class.group == group]["country"].unique().tolist()
    for year in years:
        mean_year_group = df_cost.loc[(df_cost.country.isin(group_countries)) & (df_cost.year == year),
        "pct_cost"].mean()
        df.loc[(df.country.isin(group_countries)) & (df.year == year) & (df.pct_cost.isna()),
        "pct_cost"] = mean_year_group
df = df.dropna() # no inflation data for somalia :(

##drop canada because its fucked up
df = df[df.country != 'Canada']
# and also ethiopia
df = df[~((df.country == 'Ethiopia') & (df['pop'] == 0))]

##growth rate of remittances
df = df.sort_values(by=['country', 'year'])
df['growth_rate_rem'] = df.groupby('country')['mln_euros'].pct_change() * 100  # Multiply by 100 for percentage format
df['growth_rate_pop'] = df.groupby('country')['pop'].pct_change() * 100  # Multiply by 100 for percentage format
df.replace([np.inf, -np.inf], 0, inplace=True)


for thresh in [100_00, 250_000, 500_000, 750_000, 1_000_000]:
    df["nat_dist_dummy"] = np.where(df["total affected"] > thresh, 1, 0)
    fig, ax = plt.subplots(figsize = (9,7))
    sns.scatterplot(df.loc[(df['growth_rate_rem'].abs() < 100) & (df['growth_rate_pop'].abs() < 100)], x = "growth_rate_pop", y = "growth_rate_rem", hue = 'nat_dist_dummy', ax = ax)
    plt.grid()
    fig.savefig(f"C:\\git-projects\\csh_remittances\\austria\\plots\\nat_dist_dummy_thresholds\\{thresh}_dummy.png")
    # plt.show(block = True)

##add year fixed effects

df.to_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx", index = False)

#
# ##remittances_per_migrant
# df["rem_per_migrant"] = df["mln_euros"] * 1_000_000 / df["pop"]
#
#
# fig, ax = plt.subplots(figsize=(15, 9))
# sns.regplot(df[(~df.hcpi.isna()) & (~df.rem_per_migrant.isna())], x='hcpi', y='rem_per_migrant', ax = ax, fit_reg = True)
# plt.grid(True)
# plt.title(f"hcpi v remittances sent")
# plt.xlabel('HCPI index')
# plt.ylabel('remittance sent per migrant')
# plt.show(block=True)
#
# sns.lmplot(df[(~df.hcpi.isna()) & (~df.rem_per_migrant.isna())], x='hcpi', y='rem_per_migrant', hue = 'country')
# plt.grid(True)
# plt.title(f"hcpi v remittances sent")
# plt.xlabel('HCPI index')
# plt.ylabel('remittance sent per migrant')
# plt.show(block=True)
#
#
# #####
# # cluster population evolution
# #####
# df_pop = pd.pivot_table(df, index="year", columns="country", values="pop").T
# array_pop = df_pop.to_numpy()
# time_series_pop = to_time_series_dataset(array_pop)
# scaler = TimeSeriesScalerMeanVariance()
# time_series_dataset_scaled = scaler.fit_transform(time_series_pop)
# model = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
# labels = model.fit_predict(time_series_dataset_scaled)
#
# dict_country_label = {}
# for i in range(len(df_pop.T.columns)):
#     dict_country_label[df_pop.T.columns[i]] = labels[i]
#
# df["pop_label"] = df["country"].map(dict_country_label)
# df["pop_label"] = df["pop_label"].astype(str)
#
# #diaspora population
# fig = px.scatter(df, x="pop", y="mln_euros", color = 'pop_label')
# fig.show()
#
# sns.lmplot(df_norm, x='pop', y='mln_euros', hue = 'country')
# plt.grid(True)
# plt.title(f"population v remittances sent")
# plt.xlabel('diaspora population')
# plt.ylabel('remittance sent')
# plt.legend('',frameon=False)
# plt.show(block=True)
#
# #####
# # visualise relations
# #####
#
# #diaspora population
# fig, ax = plt.subplots(figsize=(15, 9))
# sns.regplot(df, x='pop', y='mln_euros', ax = ax)
# plt.grid(True)
# plt.title(f"population v remittances sent")
# plt.xlabel('diaspora population')
# plt.ylabel('remittance sent')
# plt.show(block=True)
#
# sns.lmplot(df, x='pop', y='mln_euros', hue = 'country')
# plt.grid(True)
# plt.title(f"population v remittances sent")
# plt.xlabel('diaspora population')
# plt.ylabel('remittance sent')
# plt.legend('',frameon=False)
# plt.show(block=True)
#
# #cost of remitting
# fig, ax = plt.subplots(figsize=(15, 9))
# sns.regplot(df[df['mln_euros'] < 25], x='pct_cost', y='mln_euros', ax = ax)
# plt.grid(True)
# plt.title(f"cost of remitting v remittances sent")
# plt.xlabel('cost of remitting (%)')
# plt.ylabel('remittance sent')
# plt.show(block=True)
#
# sns.lmplot(df[df['mln_euros'] < 25], x='pct_cost', y='mln_euros', hue = 'country')
# plt.grid(True)
# plt.title(f"cost of remitting v remittances sent")
# plt.xlabel('cost of remitting (%)')
# plt.ylabel('remittance sent')
# plt.legend('',frameon=False)
# plt.show(block=True)