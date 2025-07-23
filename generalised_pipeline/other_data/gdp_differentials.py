
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import dict_names
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
from pandas.tseries.offsets import MonthEnd
from scipy.interpolate import CubicSpline


df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx")
df_gdp['date'] = pd.to_datetime(df_gdp['year'], format="%Y") + MonthEnd(0)

list_df_months = []
pbar = tqdm(df_gdp.country.unique())
for dest_country in pbar:
    pbar.set_description(f"Processing {dest_country} .. ")
    df_destination = df_gdp[df_gdp.country == dest_country].copy()
    start_date, end_date = df_destination['date'].min(), df_destination['date'].max()
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    monthly_times = (monthly_dates - start_date).days
    df_destination['time'] = (df_destination['date'] - df_destination['date'].min()).dt.days

    cols = ['date', 'destination', 'gdp']
    list_dfs = []

    try:
        data = [monthly_dates,
                [dest_country] * len(monthly_dates)]
        cs = CubicSpline(df_destination['time'],
                         df_destination["gdp"])
        vals = cs(monthly_times)
        data.append(vals)
        dict_country = dict(zip(cols, data))
        country_df = pd.DataFrame(dict_country)
        list_dfs.append(country_df)
    except:
        country_df = pd.DataFrame([])
        list_dfs.append(country_df)

    df_month = pd.concat(list_dfs)
    list_df_months.append(df_month)
result_gdp = pd.concat(list_df_months)
result_gdp = result_gdp.sort_values(["destination", "date"])
result_gdp['date'] = pd.to_datetime(result_gdp['date'])
result_gdp.to_pickle("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined.pkl")

result_gdp.to_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_splined.xlsx", index = False)

dfs = []

pbar = tqdm(df_gdp.country.unique())
for country_dest in pbar:
    pbar.set_description(f"Processing {country_dest} ...")
    df_dest = df_gdp[df_gdp.country == country_dest]
    for country_or in df_gdp.country.unique():
        df_or = df_gdp[df_gdp.country == country_or]
        merged = df_dest.merge(df_or, on = 'year', how = 'outer', suffixes=('_dest', '_or'))
        merged['gdp_diff'] = merged["gdp_dest"] - merged["gdp_or"]
        dfs.append(merged)

df_pairs = pd.concat(dfs)
df_pairs.rename(columns = {"country_dest" : "destination", "country_or" : "origin"}, inplace = True)
df_pairs = df_pairs[df_pairs.destination != df_pairs.origin]
df_pairs['date'] = pd.to_datetime(df_pairs['year'], format="%Y") + MonthEnd(0)
df_pairs.drop(columns = ["gdp_or", "gdp_dest", "year"], inplace = True)
df_pairs = df_pairs[df_pairs.date.dt.year > 2009]
df_pairs.ffill(inplace=True)

def min_max_normalize(series):
    return series / abs(series.min())

df_pairs['gdp_diff_norm'] = df_pairs.groupby('destination')['gdp_diff'].transform(min_max_normalize)

df_pairs.to_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas.pkl")