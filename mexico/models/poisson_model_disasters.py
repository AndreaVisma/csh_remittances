
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
df["prob"] = df["est_people_sending"] / df['nr_adj_with_corr_to_UN']
for state in tqdm(df.state.unique()):
    df.loc[df.state == state, 'pct_change_prob'] = (
            df.loc[df.state == state, 'prob'].pct_change() * 100)

###disaster data
emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat.Country == "Mexico") &
              (~emdat["Total Affected"].isna()) &
              (emdat["Disaster Group"] == "Natural") &
              (emdat["Start Year"] >= 2010)].copy()
emdat.sort_values("Total Affected", ascending = False, inplace = True)
# emdat = emdat[emdat["Total Affected"] >= 50_000] #more than
emdat = emdat[['Disaster Type', 'Location', 'Start Year',
       'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day', 'Total Affected']]
emdat[['Start Year','Start Month', 'Start Day']] = (
    emdat[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
emdat.rename(columns = dict(zip(['Start Year','Start Month', 'Start Day'],
                              ["year", "month", "day"])), inplace = True)
emdat["date_start"] = pd.to_datetime(emdat[["year", "month", "day"]])
emdat.drop(columns = ["year", "month", "day"], inplace = True)
emdat[['End Year','End Month', 'End Day']] = (
    emdat[['End Year','End Month', 'End Day']].fillna(12).astype(int))
emdat.rename(columns = dict(zip(['End Year','End Month', 'End Day'],
                              ["year", "month", "day"])), inplace = True)
emdat["date_end"] = pd.to_datetime(emdat[["year", "month", "day"]])
emdat.drop(columns = ["year", "month", "day"], inplace = True)

emdat['Location'] = emdat['Location'].apply(lambda x: x.split(', '))
emdat = emdat.explode('Location')
emdat.reset_index(inplace = True)

## check events that already have a nice location
nice_loc = pd.DataFrame([])
for state in tqdm(df.state.unique()):
    nice_loc_ = emdat[emdat['Location'].str.contains(state)]
    nice_loc_['Location'] = state
    nice_loc = pd.concat([nice_loc, nice_loc_])
not_nice_loc = emdat[~emdat.index.isin(nice_loc.index)].reset_index(drop = True)
nice_loc_2 = pd.DataFrame([])
for state in tqdm(dict_mex_names.keys()):
    nice_loc_ = not_nice_loc[not_nice_loc['Location'].str.contains(state)]
    nice_loc_['Location'] = dict_mex_names[state]
    nice_loc_2 = pd.concat([nice_loc_2, nice_loc_])
still_not_nice = not_nice_loc[~not_nice_loc.index.isin(nice_loc_2.index)]

emdat_states = pd.concat([nice_loc, nice_loc_2])

fig = px.scatter(emdat_states, color = 'Location', x = 'date_start', y = 'Total Affected',
                 hover_data=['Disaster Type'])
fig.show()

for state in tqdm(df.state.unique()):
    df_state = df[df.state == state].sort_values('quarter')
    fig = px.line(df_state, 'quarter', 'prob', color = 'state')
    fig.update_yaxes(title = 'Probability per quarter')
    fig.update_layout(title = "Estimated probability of a migrant sending remittances<br>"
                              "per Mexican state")
    loc = emdat_states[emdat_states.Location == state].sort_values('date_start')
    df_merge = pd.merge_asof(df_state, loc, left_on='quarter', right_on='date_start').dropna()
    df_merge = df_merge[~df_merge.duplicated('date_start')]

    fig.add_trace(go.Scatter(x = df_merge['quarter'], y = df_merge['prob'],
                             showlegend=False, hovertext=df_merge['Total Affected'],
                             mode = 'markers', marker = dict(symbol = 'x', color = 'black')))
    fig.write_html(f"c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\poisson\\states_disasters\\{state}.html")
# fig.write_html(out_folder + "poisson_probability_est.html")
# fig.show()

import seaborn as sns

fig, ax = plt.subplots(figsize = (9,6))
sns.histplot(data=df, x="prob", ax=ax)
plt.title("Probability of a migrant sending remittances per quarter,\naverage remittance of 350USD per quarter")
plt.grid(True)
ax.set_xlabel('Probability (%)')
fig.savefig("c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\poisson\\histogram_probability_all.png")
plt.show(block = True)

fig, ax = plt.subplots(figsize = (9,6))
sns.histplot(data=df, x="prob", hue = 'year', palette = 'viridis', ax=ax)
plt.title("Probability of a migrant sending remittances per quarter,\nsingle years represented")
plt.grid(True)
ax.set_xlabel('Probability (%)')
fig.savefig("c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\poisson\\histogram_probability_by_year.png")
plt.show(block = True)

fig, ax = plt.subplots(figsize = (9,6))
df[['year', 'prob']].groupby('year').mean().plot(legend = False, ax = ax)
plt.grid(True)
ax.set_xlabel('Year')
ax.set_ylabel('Probability (%)')
plt.title("Average probability of a migrant sending remittances per year")
fig.savefig("c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\poisson\\average_probability_year.png")
plt.show(block = True)

#####
# basic regression
#####

import statsmodels.api as sm

df_reg = df[['state', 'quarter', 'rem_per_quarter_USD', 'nr_adj_with_corr_to_UN']]
df_reg['rem_per_quarter_USD'] *= 1_000_000
df_reg_shift = df_reg.set_index('quarter').shift(3, freq = 'ME')
df_reg_shift = df_reg_shift.shift(1, freq = 'D').reset_index()[['state', 'quarter', 'rem_per_quarter_USD']]
df_reg_shift.rename(columns = {'rem_per_quarter_USD' : 'rem_per_quarter_t-1'}, inplace = True)
df_reg = df_reg.merge(df_reg_shift, on = ['state', 'quarter'], how = 'inner')

fig,ax = plt.subplots(figsize = (9,6))
sns.scatterplot(data=df_reg, x="nr_adj_with_corr_to_UN", y="rem_per_quarter_USD", ax = ax, hue = 'state')
plt.show(block = True)


# Outer is entity, inner is time
df_reg = df_reg.set_index(['state', 'quarter'])

from linearmodels.panel import PanelOLS
mod = PanelOLS(df_reg['rem_per_quarter_USD'], df_reg[['nr_adj_with_corr_to_UN', 'rem_per_quarter_t-1']], entity_effects=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res)

Y = df_reg['rem_per_quarter_USD']
X = df_reg[['nr_adj_with_corr_to_UN']]
X = sm.add_constant(X)
model = sm.OLS(Y,X)
res = model.fit()
print(res.summary())
