
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
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')

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

######## functions
def parse_age_group(age_group_str):
    """Helper function to parse age_group.
       This expects strings like "20-24". """
    lower, upper = map(int, age_group_str.split('-'))
    return lower, upper
def simulate_row_grouped_deterministic(row, separate_disasters=False, group_size=25):
    # Total number of agents for this row
    n_people = int(row['n_people']) // group_size

    # Get lower and upper bounds for the age group.
    lower_age, upper_age = parse_age_group(row['age_group'])

    # Simulate individual ages uniformly within the 5-year range
    # +1 in randint since upper bound is exclusive.
    ages = np.random.randint(lower_age, upper_age + 1, size=n_people)

    # Map the simulated ages to nta values using the dictionary.
    # We assume every age in the simulated sample has an entry in nta_dict.
    nta_values = np.array([nta_dict[age] for age in ages])

    # Simulate years of stay for each agent using the beta parameter.
    yrs_stay = np.random.exponential(scale=row['beta_estimate'], size=n_people).astype(int)

    # Calculate theta for each individual:
    # Here, asymmetry and gdp_diff (and even the beta from the growth rate) are constant for all individuals in the row.
    if separate_disasters:
        theta = (param_nta * (nta_values - 2)) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * (nta_values - 2)) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    total_senders = int(sum(p)) * group_size

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

def give_everyone_fixed_probability(row, separate_disasters=False, group_size=25):
    total_senders = int(row['n_people']) * 0.6

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

# now include disasters
dict_scores = dict(zip([x for x in range(12)],
                       zero_values_before_first_positive_and_after_first_negative([height + shape * np.sin((np.pi / 6) * x) for x in range(1,13)])))
def calculate_tot_score(emdat_ita, height, shape):
    global dict_scores
    dict_scores = dict(zip([x for x in range(12)],
                           zero_values_before_first_positive_and_after_first_negative(
                               [height + shape * np.sin((np.pi / 6) * x) for x in range(1, 13)])))
    for x in range(12):
        emdat_ita[f"tot_{x}"] = emdat_ita[[f"eq_{x}",f"dr_{x}",f"fl_{x}",f"st_{x}"]].sum(axis =1) * dict_scores[x]
    emdat_ita["tot_score"] = emdat_ita[[x for x in emdat_ita.columns if "tot" in x]].sum(axis =1)
    return emdat_ita[['date', 'origin', 'tot_score']]


################### run functions
param_nta = 4
param_stay = 0
param_asy = -1
param_gdp = 10
height = 0.2
shape = 0.5

def plot_individual_components(origin, destination):

    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` == '{origin}' and `destination` == '{destination}'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    for ind, row in df_nta_ita.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)

    ### no climate scores
    emdat_ita = emdat[emdat.origin.isin(df_country_ita.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape)
    df_country_ita = df_country_ita.merge(emdat_ita, on = ['origin', 'date'], how = 'left')

    df_country = df_country_ita[df_country_ita.age_group == "40-44"]
    df_plot = df_country.copy()
    df_plot['nta'] = param_nta * (nta_dict[40] - 2)
    df_plot['asymmetry'] = param_asy * df_plot["asymmetry"] * 5
    df_plot['gdp'] = param_gdp * df_plot["gdp_diff_norm"] * 5
    df_plot['tot_score'] *= 0.5
    remittance_per_period = df_plot.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")

    df_plot = df_plot[['date', 'asymmetry', 'nta', 'gdp', 'tot_score']]
    df_plot = df_plot.groupby('date').mean()
    remittance_per_period = remittance_per_period[['date', 'remittances']].groupby('date').mean()

    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot 1: Disaster Scores
    df_plot[['asymmetry', 'gdp', 'nta', 'tot_score']].plot(ax=ax)
    remittance_per_period[['remittances']].plot(ax=ax,secondary_y=True,  label='Remittances')
    ax.set_title("Theta Scores for each component")
    ax.set_ylabel("Score Value")
    ax.grid(True)
    ax.legend(loc='best')
    plt.show(block=True)

plot_individual_components(origin = "Mexico", destination = "USA")

def plot_two_countries(origin, destination):

    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` == '{origin}' and `destination` == '{destination}'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == '{destination}'""")[['age', 'nta']].fillna(0)
    for ind, row in df_nta_ita.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)

    ### no climate scores
    emdat_ita = emdat[emdat.origin.isin(df_country_ita.origin.unique())].copy()
    emdat_ita = calculate_tot_score(emdat_ita, height, shape)
    df_country_ita = df_country_ita.merge(emdat_ita, on=['origin', 'date'], how='left')

    df_country_ita['sim_senders'] = df_country_ita.apply(simulate_row_grouped_deterministic, axis=1)
    df_country_ita['sim_remittances'] = df_country_ita['sim_senders'] * 600

    remittance_per_period = df_country_ita.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()
    remittance_per_period = remittance_per_period.merge(df_rem_group, on=['date', 'origin', 'destination'], how="left")

    df_plot = remittance_per_period[['date', 'remittances', 'sim_remittances']].groupby('date').mean()

    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot 1: Disaster Scores
    df_plot[['remittances', 'sim_remittances']].plot(ax=ax)
    ax.set_title(f"{destination} to {origin}")
    ax.set_ylabel("remittances")
    ax.grid(True)
    ax.legend(loc='best')
    plt.show(block=True)

plot_two_countries(origin="Mexico", destination="USA")
