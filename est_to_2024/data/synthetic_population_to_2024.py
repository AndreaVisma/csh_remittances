"""
Script: synthetic_population_to_2024.py
Author: Generated with extension of existing process
Date: 2024
Description: 
    Extends the synthetic population estimation from 2010-2024.
    
    This script loads the pre-processed regional census-UN triangulated datasets 
    (latam, europe, row, us, germany_italy) which already contain age-sex breakdowns 
    of migrants derived from UN bilateral stocks and regional census data.
    
    For years beyond the available data in each region (e.g., Italy census only to 2021),
    this script projects forward the age-sex distribution from the latest available year.
    
    Output: Parquet file with columns:
        - date: date
        - origin: country of origin
        - destination: country of destination
        - age_group: age group (5-year bands, 0-4, 5-9, ..., 95-99)
        - sex: male or female
        - n_people: estimated number of migrants
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import sys
import os
import re
from pandas.tseries.offsets import MonthEnd

# Add project root to path for imports
sys.path.insert(0, 'C:\\git-projects\\csh_remittances')
from utils import dict_names

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================

DATA_BASE_PATH = "C:\\Data\\migration\\bilateral_stocks\\"

# Output path
OUTPUT_PATH = "C:\\Data\\migration\\bilateral_stocks\\synthetic_pop_mig_4obs.pkl"

######
# load past synthetic population estimate
df_all_old = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_3obs.pkl")

## load un bilat matrix by sex
un_file = "C:\\Data\\migration\\bilateral_stocks\\un_stock_by_sex_destination_and_origin.xlsx"

df_un = pd.read_excel(un_file, sheet_name="Table 1", skiprows=10)

df_un.rename(columns={'Region, development group, country or area of origin' : 'origin',
                   'Region, development group, country or area of destination' : 'destination'}, inplace = True)

df_un['origin'] = df_un['origin'].str.replace('*', '')
df_un = df_un[df_un.origin.isin(dict_names.keys())]
df_un['origin'] = df_un.origin.map(dict_names)

df_un['destination'] = df_un['destination'].str.replace('*', '')
df_un = df_un[df_un.destination.isin(dict_names.keys())]
df_un['destination'] = df_un.destination.map(dict_names)

df_un = df_un[["origin", "destination", "2024.1", "2024.2"]]
df_un = pd.melt(df_un, id_vars=["origin", "destination"], var_name="year", value_name="n_people")
df_un["sex"] = df_un.year.apply(lambda x: int(x[-1]))
df_un["sex"] = np.where(df_un["sex"] == 1, "male", "female")
df_un["year"] = df_un.year.apply(lambda x: int(x[:4]))

###
# load ita, germany, us files
#import GER
df_ger = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\processed_germany.pkl")
df_ger['date'] = pd.to_datetime(df_ger['date'])
df_ger = df_ger[df_ger.date == df_ger.date.max()].reset_index(drop=True)
#import ITA data
df_ita = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\italy\\processed_italy.pkl")
df_ita['date'] = pd.to_datetime(df_ita['date'])
df_ita = df_ita[df_ita.date == df_ita.date.max()].reset_index(drop=True)

#
countries = list(set(df_ita.origin).union(set(df_ger.origin)))
pbar = tqdm(countries)
for country in pbar:
    pbar.set_description(f"Processing {country} ...")
    for sex in ["male", "female"]:
        try:
            df_ger.loc[(df_ger.origin == country) & (df_ger.sex == sex), "pct"] = (
                df_ger.loc[(df_ger.origin == country) & (df_ger.sex == sex), "n_people"] /
                df_ger.loc[(df_ger.origin == country) & (df_ger.sex == sex), "n_people"].sum())
        except:
            print(f"No people from {country} in Germany")
        try:
            df_ita.loc[(df_ita.origin == country) & (df_ita.sex == sex), "pct"] = (
                df_ita.loc[(df_ita.origin == country) & (df_ita.sex == sex), "n_people"] /
                df_ita.loc[(df_ita.origin == country) & (df_ita.sex == sex), "n_people"].sum())
        except:
            print(f"No people from {country} in Italy")

df_eur = (df_ita[['date', 'origin', 'age_group', 'sex', 'n_people', 'pct']].
          merge(df_ger[['date', 'origin', 'age_group', 'sex', 'n_people', 'pct']],
                on = ['origin', 'age_group', 'sex'], how = 'outer',
                suffixes = ('_ita', '_ger')))
df_eur['diff'] = np.abs(df_eur['pct_ita'] - df_eur["pct_ger"])
df_eur.sort_values('diff', ascending = False, inplace = True)
df_eur = df_eur[(~df_eur.pct_ita.isna()) | ((~df_eur.pct_ger.isna()))]
df_eur = df_eur[df_eur.age_group != '100-104']

#import US data
df_us = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\us\\processed_usa.pkl")
df_us['date'] = pd.to_datetime(df_us['date'])
df_us = df_us[df_us.date.dt.month == 1]
df_us = df_us[df_us.date.dt.year.isin([2020])]
df_us['age'] = df_us['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_us["age_group"] = pd.cut(df_us.age, bins = bins).map(dict(zip(bins, labels)))
df_us = (df_us[['date', 'origin', 'destination', 'age_group', 'sex', 'n_people']]
      .groupby(['date', 'origin', 'destination', 'age_group', 'sex']).sum().reset_index())

#######################
# European countries
file_class = "C:\\Data\\economic\\income_classification_countries_wb.xlsx"
df_class = pd.read_excel(file_class)
europe_countries = (df_class[df_class.Region.isin(['Europe & Central Asia'])]
    ["country"].map(dict_names).unique().tolist())
europe_countries = list(set(df_un.destination).intersection(set(europe_countries)))
df_un_eur = df_un[df_un.destination.isin(europe_countries)]

#
groups, n_people_ita, n_people_ger, sexes, origins, destinations, dates = [], [], [], [], [], [], []

pbar = tqdm(europe_countries)
for dest in pbar:
    pbar.set_description(f"Processing {dest}")
    df_un_dest = df_un_eur[df_un_eur.destination == dest]
    for origin in df_un_dest['origin'].unique():
        for year in [2024]:
            for sex in ["male", "female"]:
                try:
                    tot_group = df_un_dest[(df_un_dest["sex"] == sex) &
                                           (df_un_dest.year == year) &
                                           (df_un_dest.origin == origin)].n_people.item()
                    df_ori = df_eur[(df_eur.origin == origin) & (df_eur.sex == sex)]
                    age_groups = df_ori.age_group.tolist()
                    try:
                        ratios_ita = [int(x * tot_group) for x in list(df_ori.pct_ita)]
                    except:
                        ratios_ita = [np.nan] * len(age_groups)
                    try:
                        ratios_ger = [int(x * tot_group) for x in list(df_ori.pct_ger)]
                    except:
                        ratios_ger = [np.nan] * len(age_groups)
                    groups.extend(age_groups)
                    n_people_ita.extend(ratios_ita)
                    n_people_ger.extend(ratios_ger)
                    sexes.extend([sex] * len(ratios_ita))
                    origins.extend([origin] * len(ratios_ita))
                    destinations.extend([dest] * len(ratios_ita))
                    dates.extend([year] * len(ratios_ita))
                except:
                    print(f"no vals found for origin: {origin}, dest:{dest}, period:{year}")

df_all = pd.DataFrame({"date" : dates,
                       "origin" : origins,
                       "destination" : destinations,
                       "age_group" : groups,
                       "sex" : sexes,
                       "n_people_ita" : n_people_ita,
                       "n_people_ger" : n_people_ger})

df_all['date'] = pd.to_datetime(df_all['date'], format = "%Y") + MonthEnd(13)
df_all['n_people'] = 0.5 * df_all['n_people_ita'] + 0.5 * df_all['n_people_ger']
df_all.loc[df_all.n_people_ger.isna(), 'n_people'] = df_all.loc[df_all.n_people_ger.isna(), 'n_people_ita']
df_all.loc[df_all.n_people_ita.isna(), 'n_people'] = df_all.loc[df_all.n_people_ita.isna(), 'n_people_ger']

df_all['mean_age'] = df_all['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_all["age_group"] = pd.cut(df_all.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_all.drop(columns=['n_people_ita', 'n_people_ger'], inplace=True)
df_all_eur = df_all.copy()

## LATAM
america_countries = (df_class[df_class.Region.isin(['Latin America & Caribbean', 'North America'])]
    ["country"].map(dict_names).unique().tolist())
america_countries = list(set(df_un.destination).intersection(set(america_countries)))

# slice UNDESA data to only keep LATAM+ destinations
df_un_latam = df_un[df_un.destination.isin(america_countries)]

groups, n_people, sexes, origins, destinations, dates = [], [], [], [], [], []

for dest in tqdm(america_countries):
    df_un_dest = df_un[df_un.destination == dest]
    for origin in df_un_dest['origin'].unique():
        for year in [2024]:
            for sex in ["male", "female"]:
                tot_group = df_un_dest[(df_un_dest["sex"] == sex) &
                                       (df_un_dest.year == year) &
                                       (df_un_dest.origin == origin)].n_people.item()
                df_ori = df_us[(df_us.origin == origin) & (df_us.sex == sex)]
                age_groups = df_ori.age_group.tolist()
                ratios = [int(x * tot_group) for x in list(df_ori.n_people.tolist() / df_ori.n_people.sum())]
                groups.extend(age_groups)
                n_people.extend(ratios)
                sexes.extend([sex] * len(ratios))
                origins.extend([origin] * len(ratios))
                destinations.extend([dest] * len(ratios))
                dates.extend([year] * len(ratios))

df_all = pd.DataFrame({"date" : dates,
                       "origin" : origins,
                       "destination" : destinations,
                       "age_group" : groups,
                       "sex" : sexes,
                       "n_people" : n_people})

df_all['date'] = pd.to_datetime(df_all['date'], format = "%Y") + MonthEnd(13)
df_all['mean_age'] = df_all['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_all["age_group"] = pd.cut(df_all.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_all = df_all.groupby(['date', 'origin', 'destination', 'age_group', 'sex']).agg(
    {'n_people' : 'sum', 'mean_age' : 'mean'}
).reset_index()
df_all = df_all.dropna()
df_all_america = df_all.copy()

## Rest of the World
for country in tqdm(df_us.origin.unique()):
    for date in df_us.date.unique():
        for sex in ["male", "female"]:
            df_us.loc[(df_us.origin == country) & (df_us.date == date) & (df_us.sex == sex), "pct_us"] = (
                df_us.loc[(df_us.origin == country) & (df_us.date == date) & (df_us.sex == sex), "n_people"] /
                df_us.loc[(df_us.origin == country) & (df_us.date == date) & (df_us.sex == sex), "n_people"].sum())

df_eur = df_eur.merge(df_us[['origin', 'age_group', 'sex', 'pct_us']], on = ["origin", "sex", "age_group"], how = 'outer')

### estimate world stocks
wrld_countries = (df_class[~df_class.Region.isin(['Europe & Central Asia', 'Latin America & Caribbean', 'North America'])]
    ["country"].map(dict_names).unique().tolist())
wrld_countries = list(set(df_un.destination).intersection(set(wrld_countries)))
wrld_countries = [x for x in wrld_countries if x not in ["Germany", "Italy", "USA"]]

groups, n_people_ita, n_people_ger, n_people_us, sexes, origins, destinations, dates = [], [], [], [], [], [], [], []

pbar = tqdm(wrld_countries)
for dest in pbar:
    pbar.set_description(f"Processing {dest}")
    df_un_dest = df_un[df_un.destination == dest]
    for origin in df_un_dest['origin'].unique():
        for year in [2024]:
            for sex in ["male", "female"]:
                try:
                    tot_group = df_un_dest[(df_un_dest["sex"] == sex) &
                                           (df_un_dest.year == year) &
                                           (df_un_dest.origin == origin)].n_people.item()
                    df_ori = df_eur[(df_eur.origin == origin) & (df_eur.sex == sex)]
                    age_groups = df_ori.age_group.tolist()
                    try:
                        ratios_ita = [int(x * tot_group) for x in list(df_ori.pct_ita)]
                    except:
                        ratios_ita = [np.nan] * len(age_groups)
                    try:
                        ratios_ger = [int(x * tot_group) for x in list(df_ori.pct_ger)]
                    except:
                        ratios_ger = [np.nan] * len(age_groups)
                    try:
                        ratios_us = [int(x * tot_group) for x in list(df_ori.pct_us)]
                    except:
                        ratios_us = [np.nan] * len(age_groups)
                    groups.extend(age_groups)
                    n_people_ita.extend(ratios_ita)
                    n_people_ger.extend(ratios_ger)
                    n_people_us.extend(ratios_us)
                    sexes.extend([sex] * len(ratios_ita))
                    origins.extend([origin] * len(ratios_ita))
                    destinations.extend([dest] * len(ratios_ita))
                    dates.extend([year] * len(ratios_ita))
                except:
                    print(f"no vals found for origin: {origin}, dest:{dest}, period:{year}")

df_all = pd.DataFrame({"date" : dates,
                       "origin" : origins,
                       "destination" : destinations,
                       "age_group" : groups,
                       "sex" : sexes,
                       "n_people_ita" : n_people_ita,
                       "n_people_ger" : n_people_ger,
                       "n_people_us" : n_people_us})

df_all['date'] = pd.to_datetime(df_all['date'], format = "%Y") + MonthEnd(13)

## do the average of all three where possible
df_all['n_people'] = 0.333 * df_all['n_people_ita'] + 0.333 * df_all['n_people_ger'] + 0.333 * df_all["n_people_us"]
print(df_all.n_people.isna().sum())

## where one value is missing, do average of the other two
df_all.loc[df_all.n_people_us.isna(), 'n_people'] = 0.5 * (df_all.loc[df_all.n_people_us.isna(), 'n_people_ita'] + df_all.loc[df_all.n_people_us.isna(), 'n_people_ger'])
print(df_all.n_people.isna().sum())
df_all.loc[df_all.n_people_ita.isna(), 'n_people'] = 0.5 * (df_all.loc[df_all.n_people_ita.isna(), 'n_people_us'] + df_all.loc[df_all.n_people_ita.isna(), 'n_people_ger'])
print(df_all.n_people.isna().sum())
df_all.loc[df_all.n_people_ger.isna(), 'n_people'] = 0.5 * (df_all.loc[df_all.n_people_ger.isna(), 'n_people_us'] + df_all.loc[df_all.n_people_ger.isna(), 'n_people_ger'])
print(df_all.n_people.isna().sum())

## if two values missing just take the third
df_all.loc[(df_all.n_people_us.isna()) & (df_all.n_people_ita.isna()), 'n_people'] = df_all.loc[(df_all.n_people_us.isna()) & (df_all.n_people_ita.isna()), 'n_people_ger']
print(df_all.n_people.isna().sum())
df_all.loc[(df_all.n_people_us.isna()) & (df_all.n_people_ger.isna()), 'n_people'] = df_all.loc[(df_all.n_people_us.isna()) & (df_all.n_people_ger.isna()), 'n_people_ita']
print(df_all.n_people.isna().sum())
df_all.loc[(df_all.n_people_ita.isna()) & (df_all.n_people_ger.isna()), 'n_people'] = df_all.loc[(df_all.n_people_ita.isna()) & (df_all.n_people_ger.isna()), 'n_people_us']
print(df_all.n_people.isna().sum())

assert df_all.n_people.isna().sum() == 0

df_all['mean_age'] = df_all['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_all["age_group"] = pd.cut(df_all.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_all.drop(columns=['n_people_ita', 'n_people_ger', 'n_people_us'], inplace=True)
df_all_wrld = df_all.copy()

###############
# concat everything

dfs = [df_all_old, df_all_eur, df_all_america, df_all_wrld]
df_all = pd.concat(dfs)

print(f"\nSaving to {OUTPUT_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_all.to_pickle(OUTPUT_PATH)
print("✓ Successfully saved synthetic population data")

