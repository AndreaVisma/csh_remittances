

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import random
import time
import itertools
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from random import sample
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from italy.simulation.func.goodness_of_fit import (plot_remittances_senders, plot_all_results_log,
                                                   goodness_of_fit_results, plot_all_results_lin,
                                                   plot_correlation_senders_remittances, plot_correlation_remittances)


## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[(df.date.dt.year == 2015) & (df.date.dt.month == 1)]
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
# MEX
df_rem_mex = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_renamed.xlsx")[["date", "total_mln"]]
df_rem_mex['date'] = pd.to_datetime(df_rem_mex['date'], format="%Y%m") + MonthEnd(0)
df_rem_mex['origin'] = "Mexico"
df_rem_mex['destination'] = "USA"
df_rem_mex.rename(columns = {'total_mln' : 'remittances'}, inplace = True)
df_rem_mex['remittances'] *= 1_000_000

df_rem = pd.concat([df_rem_ita, df_rem_phil, df_rem_mex])
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
def simulate_row_grouped_deterministic_probability(row, separate_disasters=False, group_size=25):
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
        theta = (param_nta * nta_values) - 2 + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * nta_values) - 2 + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score'])

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    # Simulate the remittance decision (1: sends remittance, 0: does not).
    # total_senders = int(sum(p)) * group_size

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return p

def give_everyone_fixed_probability(row, separate_disasters=False, group_size=25):
    total_senders = int(row['n_people']) * 0.6

    # Calculate the total remitted amount for this row.
    # total_remittance = total_senders * fixed_remittance * group_size
    return total_senders

################### run functions
ita_origin_countries = (df[df.destination == "Italy"]['origin'].unique().tolist())
ita_origin_countries.remove("Cote d'Ivoire")
ita_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.destination == "Italy")].origin.unique().tolist()
ita_all_countries = list(set(ita_origin_countries).intersection(set(ita_countries_high_remittances)))

phil_dest_countries = (df[df.origin == "Philippines"]['destination'].unique().tolist())
phil_countries_high_remittances = df_rem_group[(df_rem_group.remittances > 1_000_000) & (df_rem_group.origin == "Philippines")].destination.unique().tolist()
phil_all_countries = list(set(phil_dest_countries).intersection(set(phil_countries_high_remittances)))

param_nta = 2.5
param_stay = 0
param_asy = -2
param_gdp = 3
fixed_remittance_ita = df_gdp[df_gdp.origin == 'Italy'].gdp.item() * 0.01 #350  # Amount each sender sends
fixed_remittance_phil = df_gdp[df_gdp.origin == 'Japan'].gdp.item() * 0.015 #700
fixed_remittance_mex = df_gdp[df_gdp.origin == 'USA'].gdp.item() * 0.01 #800
a = 0
c = 0

########
# NTA test
df_nta['theta'] = (df_nta.nta* param_nta) -4
df_nta['prob'] = 1 / (1 + np.exp(-df_nta['theta']))
fig = px.scatter(df_nta, 'nta', 'prob')
fig.show()
#########
def produce_diaspora_profile():
    countries_ita = ita_all_countries
    countries_phil = phil_all_countries
    global nta_dict
    # df country
    df_country_ita = df.query(f"""`origin` in {countries_ita} and `destination` == 'Italy'""")
    df_country_phil = df.query(f"""`origin` == 'Philippines' and `destination` in {countries_phil}""")
    df_country_mex = df.query(f"""`origin` == 'Mexico' and `destination` == 'USA'""")

    df_country_ita = df_country_ita[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_phil = df_country_phil[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()
    df_country_mex = df_country_mex[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()

    ## nta
    df_nta_ita = df_nta.query(f"""`country` == 'Italy'""")[['age', 'nta']].fillna(0)
    df_nta_phil = df_nta[df_nta.country.isin(countries_phil)][['country', 'age', 'nta']].fillna(0)
    df_nta_mex = df_nta.query(f"""`country` == 'USA'""")[['age', 'nta']].fillna(0)

    ### no climate scores
    df_country_ita['tot_score'] = 0
    df_country_mex['tot_score'] = 0

    for ind, row in df_nta_ita.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    df_country_ita['probability'] = df_country_ita.apply(simulate_row_grouped_deterministic_probability, axis=1)
    for ind, row in df_nta_mex.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    df_country_mex['probability'] = df_country_mex.apply(simulate_row_grouped_deterministic_probability, axis=1)


    list_sims = []
    for country in tqdm(list(set(df_nta_phil.country).intersection(set(phil_all_countries)))):
        df_sim = df_country_phil[df_country_phil.destination == country].copy()
        for ind, row in df_nta_phil[df_nta_phil.country == country].iterrows():
            nta_dict[int(row.age)] = round(row.nta, 2)
        df_sim['tot_score'] = 0
        df_sim['probability'] = df_sim.apply(simulate_row_grouped_deterministic_probability, axis=1)
        list_sims.append(df_sim)
    df_country_phil = pd.concat(list_sims)


    df_country = pd.concat([df_country_ita, df_country_phil, df_country_mex])
    return df_country

df_prob = produce_diaspora_profile()

results = {}
for country in tqdm(df_prob.destination.unique()):
    df_sub = df_prob[df_prob.destination == country].copy()
    for or_ in df_sub.origin.unique():
        group = (or_, country)
        prob_list = df_sub[df_sub.origin == or_]['probability'].explode().tolist()
        results[group] = prob_list

def sample_from_dict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

# dict_to_plot = sample_from_dict(results, 4)
interest_keys = [('Switzerland', 'Italy'), ('Peru', 'Italy'), ('Mexico', 'USA'), ('Philippines', 'Japan')]
dict_to_plot = {k: results[k] for k in interest_keys}

plt.figure(figsize=(10, 6))

for label, probs in dict_to_plot.items():
    sampled_probs = [x for x in probs if x != np.nan]
    sampled_probs = np.sort(sampled_probs)
    x = np.linspace(0, 1, len(sampled_probs))
    plt.plot(x, sampled_probs, label=f"{label}", linewidth=3)

plt.title("Normalized Remittance Probability by Individual")
plt.xlabel("Normalized Individual Position (0 = first, 1 = last)")
plt.ylabel("Probability of Sending Remittances")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('C:\\Users\\Andrea Vismara\\Desktop\\papers\\remittances\\charts\\profiles_with_disasters.pdf', bbox_inches = 'tight')  # Save for PowerPoint
plt.show(block = True)