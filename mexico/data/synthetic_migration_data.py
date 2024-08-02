###

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import os
import matplotlib.ticker as mtick
import geopandas
from utils import *
import plotly.io as pio
pio.renderers.default = 'browser'

### IME data
df_ime = pd.read_excel("c:\\data\\migration\\mexico\\migrants_mex_state_aggregate.xlsx")
df_ime = df_ime[['nr_registered', 'mex_state', 'year']].groupby(['mex_state', 'year'], as_index = False).sum()
### CONAPO data
df_conapo = pd.read_csv("c:\\data\\migration\\mexico\\conapo\\iim_base2020e.csv")
df_conapo.rename(columns=dict(zip(df_conapo.columns, ['year', 'mex_state', 'tot_hh', 'hh_reciben_remesas',
                                                      'hh_with_resident_migrants_us', 'hh_with_circular_migrants_us',
                                                      'hh_with_return_migrants_us',
                                                      'ind_int_mig', 'grade_int_mig', 'rank'])), inplace = True)
df_conapo['year'] = 2020
df_conapo_2010 = pd.read_excel("c:\\data\\migration\\mexico\\conapo\\IIM2010_BASEMUN.xls")
df_conapo_2010.drop(columns = ['MUN', 'IIM0a100', 'LUG_NAL'], inplace = True)
df_conapo_2010.rename(columns=dict(zip(df_conapo_2010.columns, ['year', 'mex_state', 'municipio', 'tot_hh', 'hh_reciben_remesas',
                                                      'hh_with_resident_migrants_us', 'hh_with_circular_migrants_us',
                                                      'hh_with_return_migrants_us',
                                                      'ind_int_mig', 'grade_int_mig', 'rank'])), inplace = True)
df_conapo_2010['year'] = 2010
df_conapo_agg = df_conapo_2010[['year', 'mex_state','tot_hh']].groupby(['year', 'mex_state']).sum().reset_index()
for state in tqdm(df_conapo_agg.mex_state.unique()):
    for col in ['hh_reciben_remesas','hh_with_resident_migrants_us', 'hh_with_circular_migrants_us','hh_with_return_migrants_us']:
        df_conapo_agg.loc[df_conapo_agg.mex_state == state, col] = (df_conapo_2010.loc[df_conapo_2010.mex_state == state, 'tot_hh'] /
                                                                                     df_conapo_2010.loc[df_conapo_2010.mex_state == state, 'tot_hh'].sum()
                                                                                     * df_conapo_2010.loc[df_conapo_2010.mex_state == state, col]).sum()
df_conapo = pd.concat([df_conapo, df_conapo_agg])
df_conapo.dropna(axis = 1, inplace = True)
df_conapo['mex_state'] = df_conapo['mex_state'].map(dict_mex_names)

###merge
df = df_ime.merge(df_conapo, on=['mex_state', 'year'], how = 'outer').rename(columns = {'mex_state':'state'})
df['mig_hh'] = (0.01 * df['hh_with_resident_migrants_us'] + 0.01 * df['hh_with_circular_migrants_us']) * df.tot_hh

## check for stability
####
df_small = df[df.year.isin([2010, 2020])].copy()
df_small['year'] = df_small.year.astype('category')

outfolder = os.getcwd() + "\\mexico\\plots\\"

fig = px.scatter(df_small, x="mig_hh", y="nr_registered", hover_data=['state'],
                 trendline="ols", text="state", color='year')
fig.update_yaxes(title="Registered migrants IME")
fig.update_xaxes(title="Nr households with migrants CONAPO")
fig.update_layout(title=f'CONAPO v IME migrants data')
# fig.write_html(outfolder + f"\\IME_v_expected_migrants_2022_states.html")
fig.show()

fig = go.Figure()
for state in df_small.state.unique():
    df_state = df_small[df_small.state == state]
    fig = fig.add_trace(go.Scatter(x=df_state["mig_hh"], y=df_state["nr_registered"], text=["2010", "2020"],
                                   mode="lines+markers+text",
                               marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"), name = state))
fig.update_yaxes(title="Registered migrants IME")
fig.update_xaxes(title="Nr households with migrants")
fig.update_layout(title=f'CONAPO v IME migrants data CONAPO')
fig.write_html(outfolder + f"\\CONAPO_v_IME_migrants_2010_v_2020.html")
fig.show()
####
import statsmodels.api as sm
import numpy as np
df_small = df_small.dropna()
for year in [2010, 2020]:
    Y = df_small[df_small.year == year]["nr_registered"]
    X = df_small[df_small.year == year]["mig_hh"]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    df_small.loc[df_small.year == year, f"err"] = (df_small[df_small.year == year]["nr_registered"] -
                                                   (results.params[0] + results.params[1] * df_small[df_small.year == year]["mig_hh"]))

fig = px.scatter(df_small, x="mig_hh", y="nr_registered", hover_data=['state', 'year'],
                 trendline="ols", text = 'state')
fig.add_trace(go.Scatter(x = df_small['mig_hh'], y = df_small['err'], mode = 'markers', name = 'errors',
                         customdata= np.stack((df_small['year'], df_small['state'] ),axis = -1) ,
                         hovertemplate=
                         '<i>Error</i>:%{y:.2f}' +
                         '<br><b>mig households</b>: %{x}<br>' +
                         'year: %{customdata[0]}<br>' +
                         'state: %{customdata[1]}',
                         ))
fig.show()

fig = px.scatter(df_small, x="mig_hh", y="err", hover_data=['state', 'year'],
                 color="state", text = 'state')
fig.add_hline(y=0)
fig.show()

df_avg_err = df_small[['state', 'err']].groupby('state').mean().reset_index()
df = df[~df.state.isna()]

outfolder = os.getcwd() + "\\mexico\\plots\\IME_adjusted\\"
for state in df.state.unique():
    df.loc[df.state == state, 'nr_adj'] = df.loc[df.state == state, 'nr_registered'] - df_avg_err.loc[df_avg_err.state == state, 'err'].item()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[df.state == state].year, y=df[df.state == state].nr_registered, name='IME data'))
    fig.add_trace(go.Scatter(x = df[df.state == state].year, y = df[df.state == state].nr_adj, name = 'adjusted'))
    fig.update_layout(title = f'{state}: adjusted IME data')
    fig.write_html(outfolder + f"{state}.html")

fig = px.line(df, x = 'year', y = 'nr_adj', color='state')
fig.show()

#manual corrections
df['nr_adj_with_corr'] = df['nr_adj']
for state in ['Michoacán', 'Hidalgo', 'Zacatecas', 'Nuevo León', 'San Luis Potosí']:
    df.loc[(df.state == state) & (df.year == 2012), 'nr_adj_with_corr'] = (
            df.loc[(df.state == state) & (df.year == 2011), 'nr_adj'].item() + df.loc[(df.state == state) & (df.year == 2013), 'nr_adj'].item()) / 2
for state in tqdm(df.state.unique()):
    for year in [2014, 2019]:
        df.loc[(df.state == state) & (df.year == year), 'nr_adj_with_corr'] = (
                df.loc[(df.state == state) & (df.year == (year - 1)), 'nr_adj'].item() + df.loc[(df.state == state) & (df.year == (year + 1)), 'nr_adj'].item()) / 2

###UN data
data_folder = os.getcwd() + "\\data_downloads\\data\\"
df_all = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
                       skiprows=10, usecols="B,F,H:N", sheet_name="Table 1")
df_all.rename(columns={'Region, development group, country or area of destination': 'destination',
                   'Region, development group, country or area of origin': 'origin'}, inplace=True)
df_all['origin'] = df_all['origin'].apply(lambda x: remove_asterisc(x))
df_all['origin'] = clean_country_series(df_all['origin'])
df_all['destination'] = df_all['destination'].apply(lambda x: remove_asterisc(x))
df_all['destination'] = clean_country_series(df_all['destination'])
df_all.dropna(inplace = True)

df_mex = df_all[(df_all.origin == "Mexico") & (df_all.destination == "USA")]
df_mex = pd.melt(df_mex, id_vars=['origin', 'destination'], value_vars=[x for x in df_mex.columns[2:]],
                 var_name='year', value_name='migrants_stock')
df_mex = df_mex[df_mex.year > 1990]
df_mex.loc[len(df_mex.index) + 1] = ['Mexico', 'USA', 2022, 10_820_514]