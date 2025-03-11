
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import dict_names
import re

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

df_un = df_un[["origin", "destination", "2010.1", "2015.1", "2020.1", "2010.2", "2015.2", "2020.2"]]
df_un = pd.melt(df_un, id_vars=["origin", "destination"], var_name="year", value_name="n_people")
df_un["sex"] = df_un.year.apply(lambda x: int(x[-1]))
df_un["sex"] = np.where(df_un["sex"] == 1, "male", "female")
df_un["year"] = df_un.year.apply(lambda x: int(x[:4]))

########### load countries classification and find destination countries to process
file_class = "C:\\Data\\economic\\income_classification_countries_wb.xlsx"
df_class = pd.read_excel(file_class)
america_countries = (df_class[df_class.Region.isin(['Latin America & Caribbean', 'North America'])]
    ["country"].map(dict_names).unique().tolist())
america_countries = list(set(df_un.destination).intersection(set(america_countries)))
america_countries = [x for x in america_countries if x != "USA"]

# slice UNDESA data to only keep LATAM+ destinations
df_un = df_un[df_un.destination.isin(america_countries)]

########## load df census data
df_us = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\us\\processed_usa.pkl")
df_us = df_us[df_us.date.isin(["2010-01-31", "2015-01-31", "2020-01-31"])]

groups, n_people, sexes, origins, destinations, dates = [], [], [], [], [], []

for dest in tqdm(america_countries):
    df_un_dest = df_un[df_un.destination == dest]
    for origin in df_un_dest['origin'].unique():
        for date in ["2010-01-31", "2015-01-31", "2020-01-31"]:
            for sex in ["male", "female"]:
                tot_group = df_un_dest[(df_un_dest["sex"] == sex) &
                                       (df_un_dest.year == int(date[:4])) &
                                       (df_un_dest.origin == origin)].n_people.item()
                df_ori = df_us[(df_us.origin == origin) & (df_us.sex == sex) & (df_us.date == date)]
                age_groups = df_ori.age_group.tolist()
                ratios = [int(x * tot_group) for x in list(df_ori.n_people.tolist() / df_ori.n_people.sum())]
                groups.extend(age_groups)
                n_people.extend(ratios)
                sexes.extend([sex] * len(ratios))
                origins.extend([origin] * len(ratios))
                destinations.extend([dest] * len(ratios))
                dates.extend([date] * len(ratios))

df_all = pd.DataFrame({"date" : dates,
                       "origin" : origins,
                       "destination" : destinations,
                       "age_group" : groups,
                       "sex" : sexes,
                       "n_people" : n_people})
df_all['mean_age'] = df_all['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_all["age_group"] = pd.cut(df_all.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_all = df_all.groupby(['date', 'origin', 'destination', 'age_group', 'sex']).agg(
    {'n_people' : 'sum', 'mean_age' : 'mean'}
).reset_index()
df_all = df_all.dropna()

df_all.to_pickle("C:\\Data\\migration\\bilateral_stocks\\latam\\processed_latam_hosts_3obs.pkl")




