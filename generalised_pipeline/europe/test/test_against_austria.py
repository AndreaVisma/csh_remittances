
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import dict_names
import re
from pandas.tseries.offsets import MonthEnd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"

df_eu = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl")
df_eu = df_eu[df_eu.destination == "Austria"]
df_eu = df_eu[df_eu.date.dt.year == 2020]

df_at = pd.read_csv("C:\\Data\\migration\\austria\\pop_pyr_selected.csv")
df_at.rename(columns = {"country" : "origin", "year" : "date"}, inplace = True)
df_at["date"] = pd.to_datetime(df_at.date, format = "%Y") + MonthEnd(0)
df_at['origin'] = df_at['origin'].map(dict_names)
df_at = df_at[['date', 'age_group', 'sex', 'origin', 'n_people']].groupby(['date', 'age_group', 'sex', 'origin']).sum().reset_index()
df_at = df_at[df_at.date.dt.year == 2020]

df = df_eu[['origin', 'age_group', 'sex', 'n_people']].merge(
    df_at[['origin', 'age_group', 'sex', 'n_people']], on = ['origin', 'age_group', 'sex'],
    how = "inner", suffixes = ("_mine", "_at"))

fig1 = px.scatter(df, x = "n_people_at", y = "n_people_mine",
                  hover_data=["origin", "age_group", "sex"], color = 'origin')
fig1.add_trace(go.Scatter(
    x=list(range(int(df.n_people_at.max()))),
    y=list(range(int(df.n_people_at.max()))),
    name = "1:1 line"
))
fig1.show()
results = px.get_trendline_results(fig1)
print(results.iloc[0].item().summary())

df = df[['origin','sex', 'n_people_mine', 'n_people_at']].groupby(["origin", "sex"]).sum().reset_index()

fig1 = px.scatter(df, x = "n_people_at", y = "n_people_mine",
                  hover_data=["origin","sex"], color = 'origin')
fig1.add_trace(go.Scatter(
    x=list(range(int(df.n_people_at.max()))),
    y=list(range(int(df.n_people_at.max()))),
    name = "1:1 line"
))
fig1.show()