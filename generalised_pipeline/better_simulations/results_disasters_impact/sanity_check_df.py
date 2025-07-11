
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from utils import dict_names
import numpy as np

param_stay = 0

df_wb = pd.read_csv("C:\\Data\\remittances\\wb_remittances.txt")
df_wb = df_wb[df_wb['year'] > 2009]

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)

#########
unique_pairs = df[['origin', 'destination']].drop_duplicates()
num_unique_pairs = len(unique_pairs)
unique_countries = pd.unique(df[['origin', 'destination']].values.ravel())
N = len(unique_countries)
total_possible_pairs = N * (N - 1)
print(f"Observed unique pairs: {num_unique_pairs}")
print(f"Total possible pairs: {total_possible_pairs}")
coverage = num_unique_pairs / total_possible_pairs
print(f"Coverage of possible pairs: {coverage:.2%}")
#########

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

#########
unique_pairs_un = df_un[['origin', 'destination']].drop_duplicates()
num_unique_pairs_un = len(unique_pairs_un)
unique_countries_un = pd.unique(df_un[['origin', 'destination']].values.ravel())
N = len(unique_countries_un)
total_possible_pairs_un = N * (N - 1)
print(f"Observed unique pairs in UN data: {num_unique_pairs_un}")
print(f"Total possible pairs in UN data: {total_possible_pairs_un}")
coverage_un = num_unique_pairs_un / total_possible_pairs_un
print(f"Coverage of possible pairs in UN data: {coverage_un:.2%}")
#########

df_tot_un = df_un[['year', 'n_people']].groupby(['year']).sum()
df_tot_un_pair = df_un[['year', 'origin', 'destination', 'n_people']].groupby(['year', 'origin', 'destination']).sum()
df_tot_un['n_people'] /= 1e6
fig, ax = plt.subplots(figsize=(9, 6))
plt.plot(df_tot_un)
plt.grid(True)
plt.ylabel(f"Millions of people")
plt.title(f"Total international migrants around the world (UN data)")
plt.show(block=True)

##########################
##########################
df = df[df.origin != "Libya"]
# df = df.dropna()
df['year'] = df.date.dt.year


df_check = df[['date', 'year', 'origin', 'n_people', 'destination']].groupby(
    ['date', 'year', 'origin','destination']).sum().reset_index()
df_check_pair_year = df_check[~df_check[['year', 'origin', 'destination']].duplicated()]
df_mixed_un = df_check_pair_year.merge(df_tot_un_pair, on = ['year', 'origin', 'destination'], how = 'inner', suffixes = ("_mine", "_un"))
df_mixed_un['double_mine'] = df_mixed_un["n_people_mine"] * 2
df_tot = df[['date', 'n_people']].groupby(['date']).sum()
df_tot['n_people'] /= 1e6

fig, ax = plt.subplots(figsize=(9, 6))
plt.plot(df_tot)
plt.grid(True)
plt.ylabel(f"Millions of people")
plt.title(f"Total international migrants around the world")
plt.show(block=True)

def plot_migrants_country(origin, destination):
    df_pair = df_check[(df_check.origin == origin) & (df_check.destination == destination)][["date", "n_people"]].set_index("date")

    fig, ax = plt.subplots(figsize=(9, 6))

    plt.plot(df_pair)
    plt.grid(True)
    plt.title(f"International migrants from {origin} to {destination}")
    plt.show(block=True)

plot_migrants_country("Afghanistan", "Austria")