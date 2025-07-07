
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'
from italy.simulation.agent_based_sims.clean_simulation_procedure.load_all_data import load_data_and_param, compute_disasters_theta

from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

#######
# load all data!
#######
(params, df_ag_long, df_rem_group, df_stay_group,
 df_prob_group, gdp_group, nta_group, df) = load_data_and_param(fixed_vars =['agent_id', 'country', 'sex'])

### Define params
param_nta, param_stay, param_fam, param_gdp, rem_amount = params
eq_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dr_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
fl_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
st_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
tot_par = [0.1, 0.25, 0.5, 0.4, 0.25, 0.15, 0.1, 0, -0.1, -0.25, -0.2, -0.15, -0.1]
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [eq_par, dr_par, fl_par, st_par, tot_par]))

##########
# Effect of individual demographic characteristics
##########
def show_components_effect(country):
    df_nta_country = nta_group[nta_group.country == country][['year', 'country', 'nta']].copy()
    df_nta_country['nta'] = pd.to_numeric(df_nta_country['nta'])
    df_nta_country['nta_prob'] = 1 / (1 + np.exp(-(df_nta_country['nta'])))

    df_stay_country = df_stay_group[df_stay_group.country == country][['year', 'country', 'yrs_stay']].copy()
    df_stay_country = df_stay_country.merge(df_nta_country[['year', 'country', 'nta_prob', 'nta']], on = ['year', 'country'])
    df_stay_country['yrs_stay_eff'] = -1 * (df_stay_country['nta_prob'] - (1 / (1 + np.exp(-(
            df_stay_country['yrs_stay'] * param_stay + df_stay_country['nta'])))))

    df_fam_country = df_prob_group[df_prob_group.country == country][['year', 'country', 'fam_prob']].copy()
    df_fam_country = df_fam_country.merge(df_nta_country[['year', 'country', 'nta_prob', 'nta']], on = ['year', 'country'])
    df_fam_country['fam_eff'] = -1 * (df_fam_country['nta_prob'] - (1 / (1 + np.exp(-(
            df_fam_country['fam_prob'] * param_fam + df_fam_country['nta'])))))

    df_gdp_country = gdp_group[gdp_group.country == country][['year', 'country', 'delta_gdp_norm']].copy()
    df_gdp_country = df_gdp_country.merge(df_nta_country[['year', 'country', 'nta_prob', 'nta']], on = ['year', 'country'])
    df_gdp_country['gdp_eff'] = -1 * (df_gdp_country['nta_prob'] - (1 / (1 + np.exp(-(
            df_gdp_country['delta_gdp_norm'] * param_gdp + df_gdp_country['nta'])))))

    df_country = df_stay_country.merge(df_fam_country, on = ['country', 'year', 'nta_prob', 'nta']).merge(df_gdp_country, on = ['country', 'year', 'nta_prob', 'nta'])
    df_country = df_country.melt(id_vars=['year', 'country'],value_vars=['yrs_stay_eff', 'fam_eff', 'gdp_eff'],var_name='Effect Type',value_name='Effect Value')

    fig = px.line(df_country, x='year', y='Effect Value', color='Effect Type',title=f'{country}: Effect Trends Over the Years',
                  labels={'Effect Value': 'Effect Size', 'year': 'Year'},template='plotly_white', markers=True)
    fig.add_trace(go.Scatter(x=df_country['year'], y=[0] * len(df_country['year']), mode='lines', name='Zero Line',line=dict(color='black', dash='dash')))
    fig.show()

show_components_effect("Pakistan")
show_components_effect("Syria")

##########
# Effect of individual disasters
##########

wm = lambda x: np.average(x, weights=df.loc[x.index, "population"])
dict_dis_names = dict(zip(['eq_prob', 'dr_prob', 'st_prob', 'fl_prob', 'tot_prob'],
                          ['Earthquakes', 'Droughts', 'Storms', 'Floods', 'All disasters']))
dict_dis_colors = dict(zip(['Earthquakes', 'Droughts', 'Storms', 'Floods', 'All disasters'],
                           ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2']))
def show_disasters_effect(df, country):

    df_country = (df[(df.country == country)][["date", "country", "nta", "population", "remittances"]].groupby(["date", "country"]).
                  agg(population=("population", "sum"), remittances = ("remittances", "mean"), nta=("nta", wm))).reset_index()
    df_dis = compute_disasters_theta(df[df.country == country], dict_dis_par)

    df_country['nta'] = pd.to_numeric(df_country['nta'])
    df_country['nta_prob'] = 1 / (1 + np.exp(-(df_country['nta'])))

    ## disasters
    # earthquake
    df_country['eq_prob'] = -1 * (df_country['nta_prob'] - (1 / (1 + np.exp(-(
            np.array(df_dis['eq_score']) * 0.05 + df_country['nta'])))))
    # drought
    df_country['dr_prob'] = -1 * (df_country['nta_prob'] - (1 / (1 + np.exp(-(
            np.array(df_dis['dr_score']) * 0.05 + df_country['nta'])))))
    # storm
    df_country['st_prob'] = -1 * (df_country['nta_prob'] - (1 / (1 + np.exp(-(
            np.array(df_dis['st_score']) * 0.05 + df_country['nta'])))))
    # flood
    df_country['fl_prob'] = -1 * (df_country['nta_prob'] - (1 / (1 + np.exp(-(
            np.array(df_dis['fl_score']) * 0.05 + df_country['nta'])))))
    # all_disasters
    df_country['tot_prob'] = -1 * (df_country['nta_prob'] - (1 / (1 + np.exp(-(
            np.array(df_dis['tot_score']) * 0.05 + df_country['nta'])))))

    df_melt = df_country.melt(id_vars=['date', 'country'],
                                 value_vars=['eq_prob', 'dr_prob', 'st_prob', 'fl_prob', 'tot_prob'],
                                 var_name='Disaster Type',value_name='Disaster Impact')
    df_melt['Disaster Type'] = df_melt['Disaster Type'].map(dict_dis_names)

    window_size = 12  # Adjust this based on your data's seasonality
    df_country['remittances_trend'] = df_country['remittances'].rolling(window=window_size, center=True).mean()
    df_country['remittances_detrended'] = df_country['remittances'] - df_country['remittances_trend']

    # Compute min/max for proper alignment
    y1_min, y1_max = df_melt['Disaster Impact'].min(), df_melt['Disaster Impact'].max()
    y2_min, y2_max = df_country['remittances_detrended'].min(), df_country['remittances_detrended'].max()
    y1_range = max(abs(y1_min), abs(y1_max))
    y2_range = max(abs(y2_min), abs(y2_max))

    fig = go.Figure()
    for disaster in df_melt['Disaster Type'].unique():
        df_plot = df_melt[df_melt["Disaster Type"] == disaster]
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Disaster Impact'],mode='markers+lines',
        name=disaster, marker=dict(color=dict_dis_colors[disaster]),line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df_melt['date'], y=[0] * len(df_melt['date']), mode='lines', name='Zero Line',line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=df_country['date'],y=df_country['remittances_detrended'],name='Detrended Remittances',mode='lines',marker=dict(color='red'),yaxis='y2'))
    fig.update_layout(title=f'{country}: Disaster Effects Over the Years',
        xaxis=dict(title='Date'),yaxis=dict(title='Disaster Impact on probability', range=[-y1_range, y1_range]),
        yaxis2=dict(title='Remittances',overlaying='y',side='right', range=[-y2_range, y2_range]),template='plotly_white')
    fig.show()

show_disasters_effect(df, "Bangladesh")