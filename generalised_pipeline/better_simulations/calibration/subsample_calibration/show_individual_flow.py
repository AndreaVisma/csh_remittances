
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
import seaborn as sns
from random import sample, uniform
import random
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)
from utils import zero_values_before_first_positive_and_after_first_negative


## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df.dropna()

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

###gdp to infer remittances amount
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").rename(columns = {'country' : 'destination'})#.groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')

df['year'] = df.date.dt.year
df = df.merge(df_gdp, on=['destination', 'year'], how='left')
df['gdp'] = 0.18 * df['gdp'] / 12
## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

## load italy & Philippines remittances
# MEX
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

######## functions
def parse_age_group(age_group_str):
    """Helper function to parse age_group.
       This expects strings like "20-24". """
    lower, upper = map(int, age_group_str.split('-'))
    return lower, upper
def simulate_row_grouped_deterministic(row, separate_disasters=False):
    # Total number of agents for this row
    n_people = row['n_people']

    # Get lower and upper bounds for the age group.
    # lower_age, upper_age = parse_age_group(row['age_group'])

    # Simulate individual ages uniformly within the 5-year range
    # +1 in randint since upper bound is exclusive.
    # ages = np.random.randint(lower_age, upper_age + 1, size=n_people)

    # Map the simulated ages to nta values using the dictionary.
    # We assume every age in the simulated sample has an entry in nta_dict.
    # nta_values = np.array([nta_dict[age] for age in ages])

    # Simulate years of stay for each agent using the beta parameter.
    # yrs_stay = np.random.exponential(scale=row['beta_estimate'], size=n_people).astype(int)

    # Calculate theta for each individual:
    # Here, asymmetry and gdp_diff (and even the beta from the growth rate) are constant for all individuals in the row.
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
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    total_senders = int(p * n_people)

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

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

def calculate_tot_score(emdat_ita, height, shape, shift, months = 12):
    global dict_scores
    dict_scores = dict(zip([x for x in range(months)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * (x+shift)) for x in range(1, months + 1)])))
    for x in range(months):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]

def individual_flow(origin, dest, height, shape, shift,fixed_probability = False, plot = False, disasters = True):

    global nta_dict
    df_country_mex = df.query(f"""`origin` == '{origin}' and `destination` == '{dest}'""")

    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination', 'gdp']).mean().reset_index()

    df_nta_mex = df_nta.query(f"""`country` == '{dest}'""")[['age', 'nta']].fillna(0)

    if disasters:
        emdat_mex = emdat[emdat.origin == origin].copy()
        emdat_mex = calculate_tot_score(emdat_mex, height, shape, shift)
        df_country_mex = df_country_mex.merge(emdat_mex, on=['origin', 'date'], how='left')
    else:
        df_country_mex['tot_score'] = 0

    for ind, row in df_nta_mex.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    df_country_mex['sim_senders'] = df_country_mex.apply(simulate_row_grouped_deterministic, axis=1)
    df_country_mex['sim_remittances'] = df_country_mex['sim_senders'] * df_country_mex['gdp']

    remittance_per_period = df_country_mex.groupby(['date', 'origin', 'destination'])[['sim_remittances', 'sim_senders']].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")
    remittance_per_period['error_squared'] = (remittance_per_period['remittances'] -
                                              remittance_per_period['sim_remittances']) ** 2
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
    rem_per_period_with = individual_flow(origin, dest, height, shape, shift,
                                          fixed_probability=False, plot=False, disasters=True)
    rem_per_period_without = individual_flow(origin, dest, height, shape, shift,
                                             fixed_probability=False, plot=False, disasters=False)

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

# param_nta = 1.35
# param_asy = -7.09
# param_gdp = 9.9
# height = 0.045
# shape = 0.25
# constant = 0.13
# shift = 2

origin, dest = "Pakistan", "Italy"
plot_comparison(origin, dest)