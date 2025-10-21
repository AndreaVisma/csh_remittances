

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
from random import sample
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)

## pair of countries
origin, destination = "Philippines", "Japan"

## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)
df = df[df.n_people > 0]

##exponential betas for years of stay
df_betas = pd.read_pickle("C:\\Data\\migration\\simulations\\exponential_betas.pkl")

## family asymmetry
asymmetry_file = "C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_beginning_of_the_year.pkl"
asy_df = pd.read_pickle(asymmetry_file)

## diaspora growth rates
growth_rates = pd.read_pickle("C://data//migration//stock_pct_change.pkl")

## gdp differential
df_gdp = (pd.read_pickle("c:\\data\\economic\\gdp\\annual_gdp_deltas.pkl"))

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")


#########################################
#########################################
# Sample parameters
a = 0
c = 0
param_nta = 1
param_stay = -0.2
param_asy = -3.5
param_gdp = 0.5
fixed_remittance = 1100  # Amount each sender sends x

## load italy & Philippines remittances
df_rem_ita = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")
df_rem_ita['destination'] = 'Italy'
df_rem_ita.rename(columns = {"country": 'origin'}, inplace = True)
df_rem_ita = df_rem_ita[~df_rem_ita[["date", "origin"]].duplicated()][
    ["date", "origin", "destination", "remittances"]]
df_rem_phil = pd.read_pickle("C:\\Data\\remittances\\Philippines\\phil_remittances_detail.pkl")
df_rem = pd.concat([df_rem_ita, df_rem_phil])
df_rem['date'] = pd.to_datetime(df_rem.date)
df_rem.sort_values(['origin', 'date'], inplace=True)
df_rem_group = df_rem.copy()
df_rem_group['year'] = df_rem_group["date"].dt.year

##### disasters parameters

dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx", sheet_name="Sheet2").dropna()
df_scores = pd.read_pickle("C:\\Data\\my_datasets\\disaster_scores_only_tot.pkl")

def parse_age_group(age_group_str):
    """Helper function to parse age_group.
       This expects strings like "20-24". """
    lower, upper = map(int, age_group_str.split('-'))
    return lower, upper

def simulate_row_grouped_deterministic_probability(row, separate_disasters = False):
    # Total number of agents for this row
    n_people = int(row['n_people'])

    nta_values = np.array(nta_dict[40])

    # Simulate years of stay for each agent using the beta parameter.
    yrs_stay = 5

    # Calculate theta for each individual:
    # Here, asymmetry and gdp_diff (and even the beta from the growth rate) are constant for all individuals in the row.
    if separate_disasters:
        theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * nta_values) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    # p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    # total_senders = int(sum(p)) * group_size

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return p

def plot_individual_probability_migrants_in_italy(countries, disasters = False):
    global nta_dict
    # df country
    df_country = df.query(f"""`origin` in {countries} and `destination` == 'Italy'""")
    df_country = df_country[df_country.age_group == "40-44"]
    df_country = df_country[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country["n_people"] = 1

    # asy
    asy_df_country = asy_df.query(f"""`origin` in {countries} and `destination` == 'Italy'""")
    df_country = df_country.sort_values(['origin', 'date']).merge(asy_df_country[["date", "asymmetry", "origin"]],
                                  on=["date", "origin"], how='left').ffill()
    # growth rates
    growth_rates_cr = growth_rates.query(f"""`origin` in {countries} and`destination` == 'Italy'""")
    df_country = df_country.merge(growth_rates_cr[["date", "yrly_growth_rate", "origin"]],
                                  on=["date", "origin"], how='left')
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].bfill()
    df_country['yrly_growth_rate'] = df_country['yrly_growth_rate'].apply(lambda x: round(x, 2))
    df_country = df_country.merge(df_betas, on="yrly_growth_rate", how='left')
    ##gdp diff
    df_gdp_cr = df_gdp.query(f"""`origin` in {countries} and `destination` == 'Italy'""")
    df_country = df_country.merge(df_gdp_cr[["date", "gdp_diff_norm", "origin"]], on=["date", "origin"],
                                  how='left')
    df_country['gdp_diff_norm'] = df_country['gdp_diff_norm'].bfill()
    ## nta
    df_nta_country = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    emdat_ = df_scores[df_scores.origin.isin(countries)].copy()

    for ind, row in df_nta_country.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    dis_params['tot'] = [a, c]
    try:
        df_country.drop(columns=f"tot_score", inplace=True)
    except:
        pass
    if not disasters:
        emdat_['tot_score'] = 0
    df_country = df_country.merge(
        emdat_[(emdat_.value_a == a) & (emdat_.value_c == c)]
        [[f"tot_score", "origin", "date"]], on=["date", "origin"], how="left")
    df_country['tot_score'] = df_country['tot_score'].fillna(0)

    df_country['probability'] = df_country.apply(simulate_row_grouped_deterministic_probability, axis=1)

    fig, ax = plt.subplots(figsize = (9,6))
    for country in countries:
        df_plot = df_country[df_country.origin == country].sort_values('date')
        plt.plot(df_plot['date'], df_plot['probability'], label = f"{country}")
    plt.grid(True)
    plt.legend()
    plt.show(block = True)

plot_individual_probability_migrants_in_italy(countries = ["Mexico", "Germany", "Bangladesh"],
                                              disasters = True)