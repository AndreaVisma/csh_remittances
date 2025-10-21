
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

param_nta = 4.29
param_stay = 0
param_asy = -3.76
param_gdp = 7
height = 0.24
shape = 0.51

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
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")

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
        theta = (param_nta * (nta_values -1 )) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * (nta_values -1 )) + (param_stay * yrs_stay) \
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

def simulate_all_countries(height, shape, countries):
    global nta_dict
    df_sim = df[df.origin.isin(countries)].copy()
    df_sim = df_sim[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()

    list_res = []
    for country in tqdm(df_sim.destination.unique()):
        df_country = df_sim[df_sim.destination == country].copy()
        df_nta_country = df_nta.query(f"""`country` == '{country}'""")[['age', 'nta']].fillna(0)

        emdat_ = emdat[emdat.origin.isin(df_country.origin.unique())].copy()
        emdat_= calculate_tot_score(emdat_, height, shape)
        df_country = df_country.merge(emdat_, on = ['origin', 'date'], how = 'left').dropna()

        for ind, row in df_nta_country.iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_country['sim_senders'] = df_country.apply(simulate_row_grouped_deterministic, axis=1)
        df_country['sim_remittances'] = df_country['sim_senders'] * (df_gdp[df_gdp.origin == country].gdp.item() / 12) * 0.2
        list_res.append(df_country)

    df_country = pd.concat(list_res)
    remittance_per_period = df_country.groupby(['date', 'origin', 'destination'])['sim_remittances'].sum().reset_index()

    return remittance_per_period

#############
# countries with most disasters
dis_per_country = emdat[['origin', 'tot_0']].groupby('origin').mean().reset_index().sort_values('tot_0', ascending = False)

countries = ["Philippines", "Guatemala", "Somalia", "Chad", "China", "India", "USA", "Pakistan", "Mexico", "Haiti", "Nepal"]
dis_per_country = dis_per_country[dis_per_country.origin.isin(countries)]
#########
rem_per_period_without = simulate_all_countries(height = 0, shape = 0, countries = countries)

rem_per_period_with= simulate_all_countries(height = 0.215, shape = 0.52, countries = countries)

res = rem_per_period_without.merge(rem_per_period_with, on = ['date', 'origin', 'destination'], suffixes = ('_without', '_with'))
res = res[['origin', 'sim_remittances_without', 'sim_remittances_with']].groupby('origin').sum().reset_index()
res['difference_bn'] = (res['sim_remittances_with'] - res['sim_remittances_without']) / 1_000_000_000
res['sim_remittances_with'] /= 1_000_000_000
res['sim_remittances_without'] /= 1_000_000_000
res['pct_growth'] = 100 * (res['sim_remittances_with'] - res['sim_remittances_without']) / res['sim_remittances_without']

################
# compare with KNOMAD
import os
data_folder = os.getcwd() + "\\data_downloads\\data\\"
df_in = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')
df_in = df_in[(df_in.year >= 2010) & (df_in.year <= 2020)]
df_in['inflow'] /= 1_000
df_in = df_in[['country', 'inflow']].groupby('country').sum().reset_index()
df_in.rename(columns = {'country' : 'origin'}, inplace = True)

df_mer = res.merge(df_in, on = 'origin', how = 'left')