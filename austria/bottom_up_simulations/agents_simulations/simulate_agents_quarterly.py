"""
Script: simulate_agents_sending.py
Author: Andrea Vismara
Date: 19/11/2024
Description: simulate the remittance sending process for diasporas over the quarter
"""

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from tqdm import tqdm
from austria.bottom_up_simulations.plots.plot_results import *
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import time

## globals
fixed_vars = ['agent_id', 'country', 'sex']

## inflation correction
inflation = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx").query("Country == 'Austria' & year >= 2010")
inflation.rename(columns = {'hcpi' : 'rate'}, inplace = True)
inflation['hcpi'] = 100
for year in tqdm(inflation.year.unique()[1:]):
    inflation.loc[inflation.year == year, 'hcpi'] = (inflation.loc[inflation.year == year - 1, 'hcpi'].item() *
                                                     (1 + inflation.loc[inflation.year == year, 'rate'].item() / 100))
inflation['hcpi'] = inflation['hcpi'] / 100

# Define parameters for probability function 1
alpha = 0.2        # Minimum age to start growing the probability
beta  = 0.05  # Controls the rate of decay for age
gamma = 0.15   # Boost factor if gender is male
dict_quarter = {1 : 0, 2 : 0.07, 3 : 0, 4 : 0.06} # dictionary for quarters boosts
disaster_param = 0.002 # parameter for disasters
neighbour_param = 0.06 #parameter for neighbours
germany_param = -0.05
gdp_param = -1.4e-06
dict_group = {'Low income':-0.13, 'Lower middle income':-0.075,
              'Upper middle income':-0.09, 'High income':0.04}

# Define parameters for probability function 2
age_param = -0.001
age_param_2 = 0.1
sex_param =0.008
dict_quarter_2 = {1 : 0, 2 : 0.3, 3 : -0.11, 4 : 0.26} # dictionary for quarters boosts
disaster_param_2 = 0.0042 # parameter for disasters
neighbour_param_2 = 0.34 #parameter for neighbours
germany_param_2 = -0.05
gdp_param_2 = -0.96e-05
dict_group_2 = {'Low income':-0.61, 'Lower middle income':-0.23,
              'Upper middle income':-0.15, 'High income':0.04}

cols = ['country', 'year', 'quarter', 'total affected',
        'neighbour_dummy', 'gdp_per_capita', 'group']

## load simulated population
df = pd.read_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")
df = df[fixed_vars + [str(x) for x in range(2010, 2026)]]
df.columns = fixed_vars + [str(x) for x in range(2010, 2026)]
df = pd.melt(df, id_vars=fixed_vars, value_vars=df.columns[3:],
             value_name='age', var_name='year')
df['year'] = df['year'].astype(int) -1

## load remittances info
df_rem_quarter = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df_rem_quarter = df_rem_quarter[(df_rem_quarter.group != 0) & (df_rem_quarter.country.isin(df.country.unique().tolist()))]
for year in tqdm(df_rem_quarter.year.unique()):
    df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances'] = (df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())
df_rem_quarter['exp_population'] = df_rem_quarter['remittances'] / 450
df_rem_quarter['probability'] = df_rem_quarter['exp_population'] / df_rem_quarter['population']
####

# Define a vectorized probability function
def calculate_probability_complex(age, sex, quarter, total_affected,
                          dummy_neighbour, country, gdp_per_capita,
                          group):
    base_prob = (alpha * age ** 2 * np.e ** (-beta * age) - 18) / 70
    base_prob = max((base_prob - 0.18) / 0.5, 0)  # Adjusted to avoid negative results
    # Apply male boost if sex is male
    if sex == 'male':
        base_prob = base_prob * (1 + gamma)
    # Apply quarter adjustment
    base_prob = base_prob * (1 + dict_quarter[quarter])
    # Apply group adjustment
    base_prob = base_prob * (1 + dict_group[group])
    # Apply corrwction for natural disasters
    base_prob = base_prob * (1 + disaster_param * total_affected)
    # Apply corrwction for gdp per capita
    base_prob = base_prob * (1 + gdp_param * gdp_per_capita)
    if dummy_neighbour == 1:
        base_prob = base_prob * (1 + neighbour_param)
    if country == 'Germany':
        base_prob = base_prob * (1 + germany_param)

    return base_prob if base_prob <=1 else 1

def calculate_probability_exponential(age, sex, quarter, total_affected,
                          dummy_neighbour, country, gdp_per_capita,
                          group):
    sex_val = 1 if sex == 'male' else 0
    country_val = 1 if country == 'Germany' else 0
    base_prob = (1) / (1 + np.exp(
        age_param * age**2 + age_param_2 * age + sex_param * sex_val + disaster_param_2 * total_affected +
        dummy_neighbour * neighbour_param_2 + dict_quarter_2[quarter] + germany_param_2 * country_val +
        gdp_per_capita * gdp_param_2 + dict_group_2[group]
    ))

    return base_prob if base_prob <=1 else 1


def simulate_decisions_one_period(year, quarter):
    df_period = df[df.year == year].copy().dropna()
    df_period['quarter'] = quarter
    df_period = pd.merge(df_period, df_rem_quarter[cols],
                         on = ['country', 'year', 'quarter'], how = 'left')
    df_period.dropna(inplace=True)
    df_period['probability'] = np.vectorize(calculate_probability_exponential)(
        df_period['age'],
        df_period['sex'],
        df_period['quarter'],
        df_period['total affected'],
        df_period['neighbour_dummy'],
        df_period['country'],
        df_period['gdp_per_capita'],
        df_period['group']
    )
    df_period['decision'] = df_period['probability'].apply(lambda x: np.random.binomial(1, x, 1)[0])
    totals = df_period[['country', 'decision']].groupby('country').sum().reset_index()
    amounts_sent = dict(zip(totals.country, np.random.normal(450, 0, len(totals))))
    for country in totals.country.unique():
        totals.loc[totals.country == country, 'sim_remittances'] = totals[totals.country == country].decision.item() * amounts_sent[country]
    totals = totals.merge(
        df_rem_quarter.loc[(df_rem_quarter.year == year) & (df_rem_quarter.quarter == quarter),
        ['country', 'remittances', 'population']], on='country')
    totals['error'] = abs(totals.remittances - totals.sim_remittances)
    totals.rename(columns={'remittances': 'obs_remittances'}, inplace=True)
    totals['needed_amount_sent'] = totals['obs_remittances'] / totals['decision']
    return df_period, totals

###################
# time execution
##################
start = time.time()
df_period, totals = simulate_decisions_one_period(2022, 3)
end = time.time()
print(end - start)

########################
# one quarter simulation, check probability of sending across whole population
#######################
def probability_distribution_quarter_year(year, quarter):
    df_period, totals = simulate_decisions_one_period(year, quarter)
    df_period = df_period[['country', 'probability', 'year', 'quarter']].groupby(['country', 'year', 'quarter']).mean()
    df_period.sort_values('probability', inplace=True)
    df_period.reset_index(drop = True, inplace=True)
    #probability distribution
    fig, ax = plt.subplots()
    ax.plot(df_period['probability'], label = 'fitted values')
    ax.plot(df_rem_quarter[(df_rem_quarter.year == year) & (df_rem_quarter.quarter == quarter)].sort_values('probability').reset_index()['probability'],
            label = 'real values')
    plt.grid()
    plt.legend()
    plt.show(block = True)

probability_distribution_quarter_year(2022, 2)

def plot_needed_incomes_for_perfect_dist(year, quarter):
    df_period, totals = simulate_decisions_one_period(year, quarter)

    #amounts distribution
    fig, ax = plt.subplots()
    ax.hist(totals[totals.needed_amount_sent < 10_000]['needed_amount_sent'])
    plt.grid()
    plt.title('Distribution of perfect theoretical amounts to send')
    plt.show(block = True)
    #amounts scatter
    fig = px.scatter(totals, x='obs_remittances', y='needed_amount_sent',
                     color='country', log_x=True, log_y=False)
    fig.add_hline(450)
    fig.show()

plot_needed_incomes_for_perfect_dist(2022, 2)
###########
# one year simulations, plot results
###########
all_results = pd.DataFrame()
for quarter in tqdm([2]):
    df_period, totals = simulate_decisions_one_period(2022, quarter)
    totals['year'] = 2022
    totals['quarter'] = str(quarter)
    all_results = pd.concat([all_results, totals])
all_results['pct_population_sending'] = 100 * all_results['decision'] / all_results['population']
totals = all_results.copy()
totals['pct_population_sending'] = 100 * totals['decision'] / totals['population']

plot_all_results_log(totals)

mean_error = totals[['error', 'obs_remittances']].mean()
mean_error['relative_error'] = 100 * mean_error['error'] / mean_error['obs_remittances']
print(mean_error['relative_error'].item())

totals['pct_population_sending'].hist(bins = 20)
plt.title('Distribution of percentage of each diaspora population\nwhich is sending remittances')
plt.xlabel('Percentage of whole diaspora')
plt.ylabel('Frequency')
plt.show(block = True)

fig = px.scatter(all_results, x = 'obs_remittances', y='sim_remittances',
                 color = 'country', log_x = True, log_y = True)
fig.add_scatter(x = np.linspace(0,50_000_000, 100),
              y = np.linspace(0,50_000_000, 100))
fig.show()

############################
# iteration over parameter space
# Define parameter ranges
age_param_range = np.linspace(-0.002, 0.0, 2)  # Example range around -0.001
age_param_2_range = np.linspace(0.05, 0.15, 2)  # Example range around 0.1
sex_param_range = np.linspace(0.005, 0.01, 2)  # Example range around 0.008
disaster_param_2_range = np.linspace(0.001, 0.003, 2)  # Example range around 0.002
neighbour_param_2_range = np.linspace(0.03, 0.05, 2)  # Example range around 0.04
germany_param_2_range = np.linspace(-0.06, -0.04, 2)  # Example range around -0.05
gdp_param_2_range = np.linspace(-1.5, -1.3, 2) * 1e-05  # Example range around -1.4e0

parameter_space = list(itertools.product(
    sex_param_range,
    disaster_param_2_range,
    neighbour_param_2_range,
    germany_param_2_range
))

errors = []
for i in tqdm(range(len(parameter_space))):
    sex_param,disaster_param_2,neighbour_param_2,germany_param_2 = parameter_space[i]
    df_period, totals = simulate_decisions_one_period(2019, 3)
    mean_error = totals[['error', 'obs_remittances']].mean()
    mean_error['relative_error'] = 100 * mean_error['error'] / mean_error['obs_remittances']
    errors.append(mean_error['relative_error'].item())

plt.plot(errors)
plt.grid()
plt.show(block = True)
###########################

all_results = pd.DataFrame()
for year in tqdm([2017, 2021, 2023]):
    quarters = df_rem_quarter[df_rem_quarter.year == year].quarter.unique()
    for quarter in quarters:
        df_period, totals = simulate_decisions_one_period(year, quarter)
        totals['year'] = year
        totals['quarter'] = quarter
        all_results = pd.concat([all_results, totals])
    all_results['pct_population_sending'] = 100 * all_results['decision'] / all_results['population']

plot_all_results_log(all_results)
plot_all_results(all_results)

all_results['pct_population_sending'].hist(bins = 100)
plt.title('Distribution of percentage of each diaspora population\nwhich is sending remittances')
plt.xlabel('Percentage of whole diaspora')
plt.ylabel('Frequency')
plt.show(block = True)

mean_error = (all_results[['year', 'quarter', 'error', 'obs_remittances']].
                           groupby(['year', 'quarter']).mean())
mean_error['relative_error'] = 100 * mean_error['error'] / mean_error['obs_remittances']
mean_error['relative_error'].plot()
plt.grid()
plt.show(block = True)

fig = px.scatter(all_results, x = 'obs_remittances', y='sim_remittances',
                 color = 'country', log_x = True, log_y = True)
fig.add_scatter(x = np.linspace(0,50_000_000, 100),
              y = np.linspace(0,50_000_000, 100))
fig.show()