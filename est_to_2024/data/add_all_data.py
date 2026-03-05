


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
import itertools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated_2010_to_2024.pkl"
df = pd.read_pickle(diasporas_file)
###################
df_tot = df[["date", "origin", "destination", "n_people"]].groupby(["date", "origin", "destination"]).sum().reset_index()
df_tot = df_tot[pd.to_datetime(df_tot.date).dt.month == 12]
df_tot["date"] = pd.to_datetime(df_tot.date).dt.year
df_tot.rename(columns = {"date": "year"}, inplace = True)
df_tot.to_csv("c://data//migration//bilateral_stocks//yearly_bilateral_stock_2010_2024.csv", index = False)
###################
# df = df[df.n_people > 0]
df.loc[df.n_people < 0, 'n_people'] *= -1
df = df.sort_values(["origin", "destination", "date"])

## family asymmetry
asymmetry_file = "C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_2010_2024.pkl"
asy_df = pd.read_pickle(asymmetry_file)[['date', 'origin', 'destination', 'asymmetry']]
asy_df['date'] = pd.to_datetime(asy_df['date'])

list_df_months = []
pbar = tqdm(asy_df.destination.unique())
for dest_country in pbar:
    pbar.set_description(f"Processing {dest_country} .. ")
    df_destination = asy_df[asy_df.destination == dest_country].copy()
    start_date, end_date = df_destination['date'].min(), df_destination['date'].max()
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    monthly_times = (monthly_dates - start_date).days
    df_destination['time'] = (df_destination['date'] - df_destination['date'].min()).dt.days

    cols = ['date', 'origin', 'asymmetry']
    list_dfs = []
    for country in df_destination.origin.unique():
        try:
            data = [monthly_dates,
                    [country] * len(monthly_dates)]
            cs = CubicSpline(df_destination[(df_destination.origin == country)]['time'],
                             df_destination[(df_destination.origin == country)]["asymmetry"])
            vals = cs(monthly_times)
            data.append(vals)
            dict_country = dict(zip(cols, data))
            country_df = pd.DataFrame(dict_country)
            list_dfs.append(country_df)
        except:
            country_df = pd.DataFrame([])
            list_dfs.append(country_df)

    df_month = pd.concat(list_dfs)
    df_month['destination'] = dest_country
    list_df_months.append(df_month)

result_asy = pd.concat(list_df_months)
result_asy.loc[result_asy.asymmetry < 0, 'asymmetry'] = 0

## gdp differential
df_gdp = (pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas_2010to_2024.pkl"))
df_gdp['gdp_diff_norm'] = 2 * (df_gdp['gdp_diff'] - df_gdp['gdp_diff'].min()) / (df_gdp['gdp_diff'].max() - df_gdp['gdp_diff'].min()) - 1
df_gdp = df_gdp[["date", "origin", "destination", "gdp_diff_norm", "relative_diff"]]
df_gdp['date'] = pd.to_datetime(df_gdp['date'])

list_df_months = []
pbar = tqdm(df_gdp.destination.unique())
for dest_country in pbar:
    pbar.set_description(f"Processing {dest_country} .. ")
    df_destination = df_gdp[df_gdp.destination == dest_country].copy()
    start_date, end_date = df_destination['date'].min(), df_destination['date'].max()
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    monthly_times = (monthly_dates - start_date).days
    df_destination['time'] = (df_destination['date'] - df_destination['date'].min()).dt.days

    cols = ['date', 'origin', 'gdp_diff_norm', "relative_diff"]
    list_dfs = []
    for country in df_destination.origin.unique():
        try:
            data = [monthly_dates,
                    [country] * len(monthly_dates)]
            cs_norm = CubicSpline(df_destination[(df_destination.origin == country)]['time'],
                             df_destination[(df_destination.origin == country)]["gdp_diff_norm"])
            cs_rel = CubicSpline(df_destination[(df_destination.origin == country)]['time'],
                                  df_destination[(df_destination.origin == country)]["relative_diff"])
            vals_norm = cs_norm(monthly_times)
            vals_rel = cs_rel(monthly_times)
            data.append(vals_norm)
            data.append(vals_rel)
            dict_country = dict(zip(cols, data))
            country_df = pd.DataFrame(dict_country)
            list_dfs.append(country_df)
        except:
            country_df = pd.DataFrame([])
            list_dfs.append(country_df)

    df_month = pd.concat(list_dfs)
    df_month['destination'] = dest_country
    list_df_months.append(df_month)

result_gdp = pd.concat(list_df_months)
result_gdp = result_gdp.sort_values(["origin", "destination", "date"])
result_gdp['date'] = pd.to_datetime(result_gdp['date'])

###
df_gdp_origin = pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined_2010_to_2024.pkl")
df_gdp_origin.rename(columns = {"destination" : "origin", "gdp" : "gdp_origin"}, inplace = True)

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

### merge everything
df_sum_sexes = df[['date', 'origin', 'age_group', 'mean_age', 'destination', 'n_people']].groupby(
    ['date', 'origin', 'age_group', 'mean_age', 'destination']).sum().reset_index()
df_sum_sexes = df_sum_sexes.sort_values(["origin", "destination", "date"])
df_sum_with_asy = df_sum_sexes.merge(result_asy, on = ["date", "origin", "destination"], how = 'left')
df_asy_gdp = df_sum_with_asy.merge(result_gdp, on = ["date", "origin", "destination"], how = 'left')

nans_gdp = df_asy_gdp[(df_asy_gdp.gdp_diff_norm.isna()) & (~df_asy_gdp[["origin", "destination"]].duplicated())]
nans_asy = df_asy_gdp[(df_asy_gdp.asymmetry.isna()) & (~df_asy_gdp[["origin", "destination"]].duplicated())]

################################
df.loc[df.destination == "Cote d'Ivoire", "destination"] = "Cote dIvoire"
list_dfs = []
for country in tqdm(df.destination.unique()):
    countries_or = (df[df.destination == country]['origin'].unique().tolist())
    df_country_ita = df.query(f"""`origin` in {countries_or} and `destination` == '{country}'""")
    df_country_ita = df_country_ita[['date', 'origin', 'age_group', 'mean_age', 'destination', 'n_people']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).sum().reset_index()
    # asy
    asy_df_ita = result_asy.query(f"""`destination` == '{country}'""")
    df_country_ita = df_country_ita.sort_values(['origin', 'date']).sort_values(['origin', 'date']).merge(asy_df_ita[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill().bfill()

    ##gdp diff
    df_gdp_ita = result_gdp.query(f"""`destination` == '{country}'""")
    df_country_ita = df_country_ita.merge(df_gdp_ita[["date", "gdp_diff_norm", "relative_diff", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country_ita['gdp_diff_norm'] = df_country_ita['gdp_diff_norm'].ffill().bfill()
    df_country_ita['relative_diff'] = df_country_ita['relative_diff'].ffill().bfill()

    list_dfs.append(df_country_ita)
df_all = pd.concat(list_dfs)
df_all.loc[df_all.destination == "Cote dIvoire", "destination"] = "Cote d'Ivoire"
df_all.isna().sum()

df_all['mean_age'] = df_all['mean_age'].astype(int)
df_nta = (df_nta.rename(columns = {'age' : 'mean_age', 'country' : 'destination'})
    [['destination', 'mean_age', 'nta']])
df_all = df_all.merge(df_nta, on = ['destination', 'mean_age'], how = 'left')
df_all = df_all.merge(df_gdp_origin, on = ["date", "origin"], how = "left")

for year in tqdm(df_all.date.dt.year.unique()):

    df_all.loc[df_all.date.dt.year == year, "gdp_origin_norm"] = ((df_all.loc[df_all.date.dt.year == year, "gdp_origin"].apply(np.log) -
                                                                   df_all.loc[df_all.date.dt.year == year, "gdp_origin"].apply(np.log).min())
                                 / (df_all.loc[df_all.date.dt.year == year, "gdp_origin"].apply(np.log).max()
                                   - df_all.loc[df_all.date.dt.year == year, "gdp_origin"].apply(np.log).min()))

df_all.to_pickle("C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors_2010_to_2024.pkl")

