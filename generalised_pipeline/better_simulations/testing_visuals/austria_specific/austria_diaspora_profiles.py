
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
from utils import zero_values_before_first_positive_and_after_first_negative


## Diaspora numbers
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\interpolated_stocks_and_dem_factors.pkl"
df = pd.read_pickle(diasporas_file)
df = df[(df.date.dt.year == 2020) & (df.date.dt.month == 1)]
df = df.dropna()

##nta accounts
df_nta = pd.read_pickle("C:\\Data\\economic\\nta\\processed_nta.pkl")
nta_dict = {}

###gdp to infer remittances amount
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')

## disasters
emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags.pkl")

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
        theta = (param_nta * (nta_values -1))  + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['eq_score']) + (row['fl_score']) + (row['st_score']) + (row['dr_score'])
    else:
        theta = (param_nta * (nta_values -1)) + (param_stay * yrs_stay) \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) \
                + (row['tot_score']) + constant

    # Compute remittance probability using the logistic transformation.
    p = 1 / (1 + np.exp(-theta))
    p[nta_values == 0] = 0 # Set probability to zero if nta is zero.

    return p

################### run functions

param_nta = 4.29
param_stay = 0
param_asy = -5 #3.76
param_gdp =10 # 7
height = 0.24
shape = 0.51
constant = -2.5

aut_all_countries = (df[df.destination == "Austria"]['origin'].unique().tolist())

########

#########
def produce_diaspora_profiles_austria(disasters = False, disaster_size = 1):
    countries_aut = aut_all_countries

    global nta_dict
    # df country
    df_country_aut = df.query(f"""`origin` in {countries_aut} and `destination` == 'Austria'""")


    df_country_aut = df_country_aut[[x for x in df.columns if x != 'sex']].groupby(
        ['date', 'origin', 'age_group', 'mean_age', 'destination']).mean().reset_index()

    ## nta
    df_nta_aut = df_nta.query(f"""`country` == 'Austria'""")[['age', 'nta']].fillna(0)

    if not disasters:
        df_country_aut['tot_score'] = 0
    else:
        df_country_aut['tot_score'] = disaster_size * (height + shape * np.sin((np.pi / 6) * 3))

    for ind, row in df_nta_aut.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    df_country_aut['probability'] = df_country_aut.apply(simulate_row_grouped_deterministic_probability, axis=1)

    return df_country_aut

df_prob = produce_diaspora_profiles_austria(disasters=False)

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
interest_countries = ['Afghanistan', 'Germany', 'Turkey', 'Poland']

dict_to_plot = {(k, 'Austria'): results[(k, 'Austria')] for k in interest_countries}

fig, ax = plt.subplots(figsize=(12, 9))

for label, probs in results.items():
    sampled_probs = [x for x in probs if not np.isnan(x)]
    sampled_probs = np.sort(sampled_probs)
    x = np.linspace(0, 1, len(sampled_probs))
    plt.plot(x, sampled_probs, linewidth=2, color = 'grey', alpha = 0.3)

for label, probs in dict_to_plot.items():
    sampled_probs = [x for x in probs if not np.isnan(x)]
    sampled_probs = np.sort(sampled_probs)
    x = np.linspace(0, 1, len(sampled_probs))
    auc = np.trapz(sampled_probs, x)
    plt.plot(x, sampled_probs, label=f"{label[0]}, AUC: {round(auc, 3)}", linewidth=5)

plt.title("Normalized Remittance Probability by Individual")
plt.xlabel("Normalized Individual Position (0 = first, 1 = last)")
plt.ylabel("Probability of Sending Remittances")
# plt.legend()
plt.grid(True)
fig.savefig('.\plots\\for_paper\\diaspora_profiles_2020_AT.png', bbox_inches = 'tight')
plt.show(block = True)


################################

####
df_f = df[(df.age_group == '30-34') & (df.destination == 'Austria')]

results_dfs = []
for country in tqdm(df_f.destination.unique()):
    df_country = df_f[df_f.destination == country].copy()
    df_nta_ita = df_nta.query(f"""`country` == '{country}'""")[['age', 'nta']].fillna(0)
    for ind, row in df_nta_ita.iterrows():
        nta_dict[int(row.age)] = round(row.nta, 2)
    df_country['theta'] = (param_nta * (nta_dict[30] - 1))  \
                + (param_asy * df_country['asymmetry']) + (param_gdp * df_country['gdp_diff_norm']) + constant
    df_country['probability'] = 1 / (1 + np.exp(-df_country['theta']))
    results_dfs.append(df_country)

df_res = pd.concat(results_dfs)
df_res.sort_values('n_people', inplace = True, ascending = False)

###### plot
import seaborn as sns

sns.set_theme(style="white")
max_val_size = 0.15 * df_res.n_people.max()
interest_countries = ['Syria', 'Germany', 'Turkey', 'Poland']
min_val_interest = 0.25 * df_res[df_res.origin.isin(interest_countries)].n_people.min()

fig, ax = plt.subplots(figsize = (10,9))
df_plot_list = []
for i in range(len(interest_countries)):
    df_plot = df_res[(df_res.origin == interest_countries[i]) & (df_res.destination == 'Austria')]
    df_plot_list.append(df_plot)
df_plot = pd.concat(df_plot_list)

sns.scatterplot(x="theta", y="probability", size="n_people",
            sizes=(20, max_val_size), alpha=.4, hue = 'origin',
                data=df_res, ax = ax, palette=["grey"], legend=False)
sns.scatterplot(x="theta", y="probability", size="n_people",
            sizes=(20, max_val_size), alpha=0.8, hue = 'origin',
                data=df_plot, ax = ax, legend=False)
plt.grid(True)
# fig.savefig('.\plots\\for_paper\\individuals_probability_30yrsold_AT.png', bbox_inches = 'tight')
plt.show(block = True)

################ average