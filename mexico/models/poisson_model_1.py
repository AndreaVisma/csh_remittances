

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
from tqdm import tqdm
pio.renderers.default = 'browser'

out_folder = "c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\poisson\\"

##
df_ime = pd.read_excel("c:\\data\\migration\\mexico\\migrants_per_state_IME_adj.xlsx")

df_bdm = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_state_long.xlsx")
df_bdm.rename(columns = {df_bdm.columns[0]: "quarter"}, inplace = True)
df_bdm["year"] = df_bdm.quarter.dt.year

df = df_ime.merge(df_bdm, on = ['year', 'state'])

### estimate probabilities with fixed remittances assumption
avg_rem = 350
df["rem_per_quarter_USD"] = df["mln_USD_remesas"] * 1_000_000
df["est_people_sending"] = df["rem_per_quarter_USD"] / avg_rem

fig = px.line(df.sort_values(['state', 'quarter']), 'quarter', 'est_people_sending', color = 'state')
fig.update_layout(title = "Estimated people sending remittances per Mexican state")
fig.write_html(out_folder + "est_people_sending.html")
fig.show()

df["poisson_prob"] = df["est_people_sending"] / df['nr_adj_with_corr_to_UN']

fig = px.line(df, 'quarter', 'poisson_prob', color = 'state')
fig.update_yaxes(title = 'Poisson probability per quarter')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "per Mexican state")
fig.write_html(out_folder + "poisson_probability_est.html")
fig.show()

for state in tqdm(df.state.unique()):
    df.loc[df.state == state, 'pct_change_poisson_prob'] = (
            df.loc[df.state == state, 'poisson_prob'].pct_change() * 100)

fig = px.line(df, 'quarter', 'pct_change_poisson_prob', color = 'state')
fig.update_yaxes(title = 'Percentage change in the Poisson probability per quarter')
fig.update_layout(title = "Percentage change in the Poisson probability of a migrant sending remittances<br>"
                          "per Mexican state")
fig.write_html(out_folder + "poisson_probability_change.html")
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
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_probability_whole_Mexico_quarter.html")
fig.show()

df_year["poisson_prob_change"] = df_year["poisson_prob"].pct_change() * 100

fig = px.line(df_year, 'date', 'poisson_prob_change')
fig.update_yaxes(title = 'Percentage change in Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_probability_change_whole_Mexico_quarter.html")
fig.show()

df_year = (df_year[['year', 'total_mln_seas','rem_per_month_USD', 'est_people_sending', 'nr_adj_with_corr_to_UN']]
           .groupby('year').
agg({'total_mln_seas' : 'sum', 'rem_per_month_USD' : 'mean', 'est_people_sending' : 'mean',
    'nr_adj_with_corr_to_UN' : 'mean'
})).reset_index()
df_year["poisson_prob"] = df_year["est_people_sending"] / df_year['nr_adj_with_corr_to_UN']

fig = px.line(df_year, 'year', 'poisson_prob')
fig.update_yaxes(title = 'Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_probability_change_whole_Mexico_year.html")
fig.show()

df_year["poisson_prob_change"] = df_year["poisson_prob"].pct_change() * 100

fig = px.line(df_year, 'year', 'poisson_prob_change')
fig.update_yaxes(title = 'Percentage change in Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "change_poisson_probability_change_whole_Mexico_year.html")
fig.show()

#####estimate probabilities with changing average
df_ime_year = df_ime[['year', 'nr_adj_with_corr_to_UN']].groupby('year').sum().reset_index()
df_rem = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_seasonally_adjusted.xlsx")
df_rem = df_rem[['date', 'total_mln_seas', 'total_mean_op_seas']]
df_rem['year'] = df_rem.date.dt.year

df_year = df_ime_year.merge(df_rem, on = ['year'])
df_year["rem_per_month_USD"] = df_year["total_mln_seas"] * 1_000_000
df_year["est_people_sending"] = df_year["rem_per_month_USD"] / df_year['total_mean_op_seas']

fig = px.line(df_year, 'date', 'est_people_sending')
fig.show()

df_year["poisson_prob"] = df_year["est_people_sending"] / df_year['nr_adj_with_corr_to_UN']

fig = px.line(df_year, 'date', 'poisson_prob')
fig.update_yaxes(title = 'Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_prob_whole_Mexico_quarter_CHANNGING_AVG.html")
fig.show()

df_year["poisson_prob_change"] = df_year["poisson_prob"].pct_change() * 100

fig = px.line(df_year, 'date', 'poisson_prob_change')
fig.update_yaxes(title = 'Percentage change in Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability change of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_probability_change_whole_Mexico_quarter_CHANGING_AVG.html")
fig.show()

df_year = (df_year[['year', 'total_mln_seas','rem_per_month_USD', 'est_people_sending', 'nr_adj_with_corr_to_UN']]
           .groupby('year').
agg({'total_mln_seas' : 'sum', 'rem_per_month_USD' : 'mean', 'est_people_sending' : 'mean',
    'nr_adj_with_corr_to_UN' : 'mean'
})).reset_index()
df_year["poisson_prob"] = df_year["est_people_sending"] / df_year['nr_adj_with_corr_to_UN']

fig = px.line(df_year, 'year', 'poisson_prob')
fig.update_yaxes(title = 'Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_probability_whole_Mexico_year_CHANGING_AVG.html")
fig.show()

df_year["poisson_prob_change"] = df_year["poisson_prob"].pct_change() * 100

fig = px.line(df_year, 'year', 'poisson_prob_change')
fig.update_yaxes(title = 'Percentage change in Poisson probability per month')
fig.update_layout(title = "Estimated Poisson probability of a migrant sending remittances<br>"
                          "whole of Mexico")
fig.write_html(out_folder + "poisson_probability_change_whole_Mexico_year_CHANGING_AVG.html")
fig.show()