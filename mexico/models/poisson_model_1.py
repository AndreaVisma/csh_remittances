

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import os
import matplotlib.ticker as mtick
import geopandas
from utils import *
import plotly.io as pio
pio.renderers.default = 'browser'

out_folder = "c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\poisson\\"

##
df_ime = pd.read_excel("c:\\data\\migration\\mexico\\migrants_per_state_IME_adj.xlsx")

df_bdm = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_state_long.xlsx")
df_bdm.rename(columns = {df_bdm.columns[0]: "quarter"}, inplace = True)
df_bdm["year"] = df_bdm.quarter.dt.year

df = df_ime.merge(df_bdm, on = ['year', 'state'])

### estimate probabilities
avg_rem = 350
df["rem_per_quarter_USD"] = df["mln_USD_remesas"] * 1_000_000
df["est_people_sending"] = df["rem_per_quarter_USD"] / avg_rem

fig = px.line(df, 'quarter', 'est_people_sending', color = 'state')
fig.show()

df["poisson_prob"] = df["est_people_sending"] / df['nr_adj_with_corr_to_UN']

fig = px.line(df, 'quarter', 'poisson_prob', color = 'state')
fig.update_yaxes(title = 'Poisson probability per quarter')
fig.show()

##national monthly level
df_ime_year = df_ime[['year', 'nr_adj_with_corr_to_UN']].groupby('year').sum().reset_index()
df_rem = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_seasonally_adjusted.xlsx")
df_rem = df_rem[['date', 'total_mln_seas']]
df_rem['year'] = df_rem.date.dt.year

df_year = df_ime_year.merge(df_rem, on = ['year'])

### estimate probabilities
avg_rem = 350
df_year["rem_per_month_USD"] = df_year["total_mln_seas"] * 1_000_000
df_year["est_people_sending"] = df_year["rem_per_month_USD"] / avg_rem

fig = px.line(df_year, 'date', 'est_people_sending')
fig.show()

df_year["poisson_prob"] = df_year["est_people_sending"] / df_year['nr_adj_with_corr_to_UN']

fig = px.line(df_year, 'date', 'poisson_prob')
fig.update_yaxes(title = 'Poisson probability per month')
fig.show()