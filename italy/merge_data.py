"""
Script: merge_data.py
Author: Andrea Vismara
Date: 11/02/2024
Description: merge all the data we have to make it ready for the simulation process
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

#remittances
df_rem = pd.read_csv('c:\\data\\remittances\\italy\\monthly_splined_remittances.csv')
df_rem['date'] = pd.to_datetime(df_rem['date'])
df_rem['year'] = df_rem['date'].dt.year

## disasters
disasters = ['Drought', 'Earthquake', 'Flood', 'Storm']
disasters_short = ['dr', 'eq', 'fl', 'st']
disaster_names = dict(zip(disasters, disasters_short))

df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_nat = df_nat[df_nat.type.isin(disasters)]
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]

df_nat = df_nat.pivot_table(index=['week_start', 'country'], columns='type', values='total_affected', aggfunc = 'sum')
df_nat = df_nat.reset_index()
df_nat_monthly = (df_nat.groupby(['country', pd.Grouper(key='week_start', freq='M')]).sum()
    .reset_index() .rename(columns={'week_start': 'date'}))
df_nat_monthly.rename(columns = disaster_names, inplace=True)

# shift disasters
df_nat_monthly['date'] = pd.to_datetime(df_nat_monthly['date'])

### remittances
df_rem = df_rem.merge(df_nat_monthly, on = ["country", "date"], how = 'left')
df_rem.fillna(0, inplace = True)
df_rem.sort_values(['country', 'date'], inplace=True)
# Create shifted columns using proper datetime handling
for col in disasters_short:
    for shift in tqdm([int(x) for x in np.linspace(1, 12, 12)]):
        g = df_rem.groupby('country', group_keys=False)
        g =  g.apply(lambda x: x.set_index('date')[col]
                     .shift(shift).reset_index(drop=True)).fillna(0)
        df_rem[f'{col}_{shift}'] = g.tolist()
df_rem['tot'] = df_rem['fl'] + df_rem['eq'] + df_rem['st'] + df_rem['dr']
for shift in tqdm([int(x) for x in np.linspace(1, 12, 12)]):
    df_rem[f'tot_{shift}'] = df_rem[f'fl_{shift}'] + df_rem[f'eq_{shift}'] + df_rem[f'st_{shift}'] + df_rem[f'dr_{shift}']

df_rem.to_parquet("C:\\Data\\my_datasets\\italy\\remittances_disasters_data.parquet")

# population and national transfers account
df = pd.read_csv('c:\\data\\migration\\italy\\estimated_stocks_new.csv')
df['age'] = df.age_group.astype(str).apply(lambda x: np.mean(list(map(int, re.findall(r'\d+', x)))))
df.loc[df.age == 5, 'age'] = 2.5
df['age'] = df.age.astype(int)
df.sort_values(['citizenship', 'year', 'age'], inplace = True)

nta = pd.read_excel("c:\\data\\economic\\nta\\NTA profiles.xlsx", sheet_name="italy").T
nta.columns = nta.iloc[0]
nta = nta.iloc[1:]
nta.reset_index(names='age', inplace = True)
nta = nta[['age', 'Support Ratio']].rename(columns = {'Support Ratio' : 'nta'})
# nta.nta=(nta.nta-nta.nta.min())/(nta.nta.max()-nta.nta.min()) - 0.15
nta.loc[nta.nta <0, 'nta'] = 0

df = df.merge(nta, on='age', how = 'left')
df.rename(columns = {'citizenship' : 'country', 'count' : 'population'}, inplace = True)
df = df.merge(df_rem, on = ['country', 'year'], how = 'left')
df.dropna(inplace = True)
df.loc[df.population < 0, 'population'] = 0
# df.drop(columns = 'remittances', inplace = True)

##add gdp
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\quarterly_gdp_clean.xlsx")
for year in tqdm(df_gdp.year.unique(),
                          total = len(df_gdp.year.unique())):
    df_year = df_gdp[df_gdp.year == year]
    for quarter in df_year.quarter.unique():
        df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter), 'delta_gdp'] = (
                df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter), 'gdp_per_capita'] -
                df_gdp.loc[(df_gdp.year == year) & (df_gdp.quarter == quarter) & (df_gdp.country == 'Austria'), 'gdp_per_capita'].item())
df_gdp = df_gdp[['country', 'year', 'gdp_per_capita', 'delta_gdp']].groupby(['country', 'year']).mean().reset_index()
df = df.merge(df_gdp, on = ['country', 'year'], how = 'left')
df.dropna(inplace = True)

df.to_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
