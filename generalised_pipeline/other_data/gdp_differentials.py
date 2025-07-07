
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import dict_names
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
from pandas.tseries.offsets import MonthEnd

df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx")

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

# df_ag_long["delta_gdp_norm"] = df_ag_long.delta_gdp / abs(df_ag_long.delta_gdp.min())
def min_max_normalize(series):
    return series / abs(series.min())

df_pairs['gdp_diff_norm'] = df_pairs.groupby('destination')['gdp_diff'].transform(min_max_normalize)

df_pairs.to_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas.pkl")