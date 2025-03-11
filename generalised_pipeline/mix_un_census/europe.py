

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import dict_names
import re
from pandas.tseries.offsets import MonthEnd

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
europe_countries = (df_class[df_class.Region.isin(['Europe & Central Asia'])]
    ["country"].map(dict_names).unique().tolist())
europe_countries = list(set(df_un.destination).intersection(set(europe_countries)))
europe_countries = [x for x in europe_countries if x not in ["Germany", "Italy"]]

# slice UNDESA data to only keep EUROPE destinations
df_un = df_un[df_un.destination.isin(europe_countries)]

df_eur = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\germany\\germany_italy_ratios.pkl")
df_eur['date'] = pd.to_datetime(df_eur['date'])
df_eur = df_eur[df_eur.date.dt.month == 1]

groups, n_people_ita, n_people_ger, sexes, origins, destinations, dates = [], [], [], [], [], [], []

for dest in tqdm(europe_countries, desc="Processing ..."):
    df_un_dest = df_un[df_un.destination == dest]
    for origin in df_un_dest['origin'].unique():
        for year in [2010, 2015, 2020]:
            for sex in ["male", "female"]:
                try:
                    tot_group = df_un_dest[(df_un_dest["sex"] == sex) &
                                           (df_un_dest.year == year) &
                                           (df_un_dest.origin == origin)].n_people.item()
                    df_ori = df_eur[(df_eur.origin == origin) & (df_eur.sex == sex) & (df_eur.date.dt.year == year)]
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

df_all['date'] = pd.to_datetime(df_all['date'], format = "%Y") + MonthEnd(0)
df_all['n_people'] = 0.5 * df_all['n_people_ita'] + 0.5 * df_all['n_people_ger']
df_all.loc[df_all.n_people_ger.isna(), 'n_people'] = df_all.loc[df_all.n_people_ger.isna(), 'n_people_ita']
df_all.loc[df_all.n_people_ita.isna(), 'n_people'] = df_all.loc[df_all.n_people_ita.isna(), 'n_people_ger']

df_all['mean_age'] = df_all['age_group'].apply(lambda x: np.mean([int(y) for y in re.findall(r'\d+', x)]))
bins = pd.interval_range(start=0, periods=20, end = 100, closed = "left")
labels = [f"{5*i}-{5*i+4}" for i in range(20)]
df_all["age_group"] = pd.cut(df_all.mean_age, bins = bins).map(dict(zip(bins, labels)))
df_all.drop(columns=['n_people_ita', 'n_people_ger'], inplace=True)

df_all.to_pickle("C:\\Data\\migration\\bilateral_stocks\\europe\\processed_european_hosts_3obs.pkl")
