

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
from random import sample, uniform
import random
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)
from utils import zero_values_before_first_positive_and_after_first_negative

param_stay = 0

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.origin != "Libya"]
df = df.dropna()
df['year'] = df.date.dt.year

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

df['mean_age'] = df['mean_age'].astype(int)
for country in tqdm(df.destination.unique()):
    for ind, row in df_nta[df_nta.country == country].iterrows():
        nta_dict[int(row.age)] =row.nta
    df.loc[df.destination == country, 'nta'] = df.loc[df.destination == country, 'mean_age'].map(nta_dict)

###gdp to infer remittances amount
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns = {'country' : 'destination'})#.groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')
df = df.merge(df_gdp, on=['destination', 'year'], how='left')
df['rem_amount'] = 0.18 * df['gdp'] / 12

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

## load italy & Philippines remittances
#ITA
df_rem_ita = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df_rem_ita['destination'] = 'Italy'
df_rem_ita.rename(columns = {"country": 'origin'}, inplace = True)
df_rem_ita = df_rem_ita[~df_rem_ita[["date", "origin"]].duplicated()][
    ["date", "origin", "destination", "remittances"]]
# PHIL
df_rem_phil = pd.read_pickle("C:\\Data\\remittances\\Philippines\\phil_remittances_detail.pkl")
# PAK
df_rem_pak = pd.read_pickle("C:\\Data\\remittances\\Pakistan\\pak_remittances_detail.pkl")
# GUA
df_rem_gua = pd.read_pickle("C:\\Data\\remittances\\Guatemala\\gua_remittances_detail.pkl")
# GUA
df_rem_nic = pd.read_pickle("C:\\Data\\remittances\\Nicaragua\\nic_remittances_detail.pkl")
# MEX
df_rem_mex = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_renamed.xlsx")[["date", "total_mln"]]
df_rem_mex['date'] = pd.to_datetime(df_rem_mex['date'], format="%Y%m") + MonthEnd(0)
df_rem_mex['origin'] = "Mexico"
df_rem_mex['destination'] = "USA"
df_rem_mex.rename(columns = {'total_mln' : 'remittances'}, inplace = True)
df_rem_mex['remittances'] *= 1_000_000

df_rem = pd.concat([df_rem_ita, df_rem_phil, df_rem_mex, df_rem_pak, df_rem_gua, df_rem_nic])
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem.sort_values(['origin', 'date'], inplace=True)
df_rem_group = df_rem.copy()
df_rem_group['year'] = df_rem_group["date"].dt.year

df = df.merge(df_rem, on =['date', 'origin', 'destination'], how = 'left')
df.dropna(inplace = True)

######## functions

def simulate_row_grouped_deterministic(row, separate_disasters=False):
    # Total number of agents for this row
    n_people = row['n_people']

    if row["nta"] != 0:
        if separate_disasters:
            theta = constant + (param_nta * (row['nta'])) \
                    + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                    + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
        else:
            theta = constant + (param_nta * (row['nta'])) \
                    + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                    + (row['tot_score'])
        # Compute remittance probability using the logistic transformation.
        p = 1 / (1 + np.exp(-theta))
    else:
        p = 0

    total_senders = int(p * n_people)

    return total_senders

################### run functions
def get_df_countries(df):
    ita_origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
    try:
        ita_origin_countries.remove("Cote d'Ivoire")
    except:
        pass
    ita_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.destination == "Italy")].origin.unique().tolist()
    ita_all_countries = list(set(ita_origin_countries).intersection(set(ita_countries_high_remittances)))

    phil_dest_countries = (df[df.origin == "Philippines"]['destination'].unique().tolist())
    phil_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Philippines")].destination.unique().tolist()
    phil_all_countries = list(set(phil_dest_countries).intersection(set(phil_countries_high_remittances)))

    pak_dest_countries = (df[df.origin == "Pakistan"]['destination'].unique().tolist())
    pak_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Pakistan")].destination.unique().tolist()
    pak_all_countries = list(set(pak_dest_countries).intersection(set(pak_countries_high_remittances)))

    nic_dest_countries = (df[df.origin == "Nicaragua"]['destination'].unique().tolist())
    nic_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Nicaragua")].destination.unique().tolist()
    nic_all_countries = list(set(nic_dest_countries).intersection(set(nic_countries_high_remittances)))

    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    countries_pak = pak_all_countries
    countries_nic = nic_all_countries

    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_pak = df.query(f"""`origin` == 'Pakistan' and `destination` in {countries_pak}""")
    df_country_mex = df.query(f"""`origin` == 'Mexico' and `destination` == 'USA'""")
    df_country_nic = df.query(f"""`origin` == 'Nicaragua' and `destination` in {countries_nic}""")
    df_country_gua = df.query(f"""`origin` == 'Guatemala' and `destination` == 'USA'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year', 'origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_pak = df_country_pak[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_nic = df_country_nic[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()
    df_country_gua = df_country_gua[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'year','origin', 'age_group', 'mean_age', 'destination', 'gdp', 'rem_amount', 'nta']).mean().reset_index()

    df_countries = pd.concat([df_country_ita, df_country_mex, df_country_gua, df_country_nic, df_country_pak, df_country_phil])
    return df_countries

def plot_country_mean(df, two_countries = False):
    if two_countries:
        df_mean_ita = df[df.destination == 'Italy'][['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        df_mean_phil = df[df.origin == 'Philippines'][['destination', 'remittances', 'sim_remittances']].groupby(['destination']).mean().reset_index()
        df_mean_pak = df[df.origin == 'Pakistan'][['destination', 'remittances', 'sim_remittances']].groupby(
            ['destination']).mean().reset_index()
        df_mean_nic = df[df.origin == 'Nicaragua'][['destination', 'remittances', 'sim_remittances']].groupby(
            ['destination']).mean().reset_index()
        df_mean_mex = df[df.origin.isin(["Mexico", "Guatemala"])][['origin', 'remittances', 'sim_remittances']].groupby(
            ['origin']).mean().reset_index()
        df_mean = pd.concat([df_mean_ita, df_mean_phil, df_mean_mex, df_mean_pak, df_mean_nic])
        fig = go.Figure()

        # Add traces with loop
        for df, color, name, text_col, prefix in zip(
                [df_mean_ita, df_mean_phil, df_mean_mex, df_mean_pak, df_mean_nic],
                ['blue', 'red', 'orange', 'green', 'pink'],
                ['From Italy', 'To Philippines', 'from USA', 'To Pakistan', 'To Nicaragua'],
                ['origin', 'destination', 'origin', 'destination', 'destination'],
                ['Origin', 'Destination', 'origin', 'destination', 'destination']
        ):
            fig.add_trace(go.Scatter(
                x=df['remittances'],
                y=df['sim_remittances'],
                mode='markers',
                name=name,
                marker=dict(color=color, size = 10),
                text=df[text_col],
                hovertemplate=f'{prefix}: %{{text}}<br>Remittances: %{{x}}<br>Simulated: %{{y}}'
            ))

        # Add 1:1 line
        max_val = max(df_mean_ita['remittances'].max(), df_mean_phil['remittances'].max(), df_mean_mex['remittances'].max())
        fig.add_trace(go.Scatter(
            x=np.linspace(0, max_val, 100),
            y=np.linspace(0, max_val, 100),
            mode='lines',
            name='1:1 Line',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title='Simulated vs Actual Remittances',
            xaxis=dict(title='Actual Remittances (log scale)'),
            yaxis=dict(title='Simulated Remittances (log scale)'),
            legend=dict(title='Legend'),
            template='plotly_white'
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    else:
        df_mean = df[['origin', 'remittances', 'sim_remittances']].groupby(['origin']).mean().reset_index()
        fig = px.scatter(df_mean, x = 'remittances', y = 'sim_remittances',
                         color = 'origin', log_x=True, log_y=True)
        fig.add_scatter(x=np.linspace(0, df_mean.remittances.max(), 100),
                        y=np.linspace(0, df_mean.remittances.max(), 100))
    fig.show()
    goodness_of_fit_results(df_mean)

def calculate_tot_score(emdat_ita, height, shape, shift):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def check_params_combo(df_countries, height, shape, shift, rem_pct, plot = True):

    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns = 'tot_score', inplace = True)
    except:
        pass
    df_countries = df_countries.merge(emdat_ita, on=['origin', 'date'], how='left')
    df_countries['rem_amount'] = rem_pct * df_countries['gdp'] / 12

    df_countries['sim_senders'] = df_countries.apply(simulate_row_grouped_deterministic, axis=1)
    df_countries['sim_remittances'] = df_countries['sim_senders'] * df_countries['rem_amount']

    remittance_per_period = df_countries.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")

    if plot:
        goodness_of_fit_results(remittance_per_period)

        plot_country_mean(remittance_per_period, two_countries=True)

    return remittance_per_period

def individual_flow(df_countries, origin, dest, height, shape, shift, rem_pct, plot = True, disasters = True):

    emdat_ita = emdat[emdat.origin.isin(df_countries.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape, shift)
    try:
        df_countries.drop(columns='tot_score', inplace=True)
    except:
        pass
    df_country = df_countries[(df_countries.origin == origin) & (df_countries.destination == dest)].copy()
    if disasters:
        df_country = df_country.merge(emdat_ita, on=['origin', 'date'], how='left')
    else:
        df_country['tot_score'] = 0
    df_country['rem_amount'] = rem_pct * df_country['gdp'] / 12

    df_country['sim_senders'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
    df_country['sim_remittances'] = df_country['sim_senders'] * df_country['rem_amount']

    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])[
        ['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    if plot:
        to_plot = remittance_per_period[['date', 'sim_remittances', 'remittances']].copy()
        to_plot.set_index('date', inplace = True)
        to_plot.plot()
        plt.title(f"Remittances from {dest} to {origin}")
        plt.grid(True)
        plt.show(block = True)

        goodness_of_fit_results(remittance_per_period)

    return remittance_per_period

########
def plot_comparison(origin, dest):
    rem_per_period_with = individual_flow(df_countries, origin, dest,
                            height, shape, shift, rem_pct, plot = False, disasters = True)

    rem_per_period_without = individual_flow(df_countries, origin, dest,
                                height, shape, shift, rem_pct, plot = False, disasters = False)

    to_plot = rem_per_period_with[['date', 'remittances', 'sim_remittances']].merge(
        rem_per_period_without[['date', 'sim_remittances']],
        on='date', suffixes=('_with', '_without')).set_index('date')
    to_plot.plot()
    plt.title(f"Remittances from {dest} to {origin}")
    plt.grid(True)
    plt.show(block=True)

    to_plot['difference'] = to_plot['sim_remittances_with'] - to_plot['sim_remittances_without']
    print(f"Extra remittances: {to_plot['difference'].sum() / 1e9} bn euros")
    print(f"Pct increase: {100 * to_plot['difference'].sum() / to_plot['sim_remittances_without'].sum()}%")


#######
df_countries = get_df_countries(df)
params = [2.66093400319252, -9.771767189034518, 8.310441847102053,
                                    0.338599981797477, 0.3901969522813624,-0.750105008323315,
                                    0.6917237435288245, 0.1942257641159886]
param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct = params

origin, dest = "Mexico", "Italy"
plot_comparison(origin, dest)