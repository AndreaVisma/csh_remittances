"""
Script: simulate_agents_optimised.py
Author: Andrea Vismara
Date: 26/11/2024
Description: simulate the remittance sending process for diasporas over the quarter
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import time
import itertools
import matplotlib.pyplot as plt
from austria.bottom_up_simulations.plots.plot_results import *
from austria.bottom_up_simulations.plots.goodness_of_fit_func import goodness_of_fit_results
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
from utils import europe

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

# Define parameters for probability function
####################
amount_sent = 450
deviation_money = 0
####################
# run_sims(2019)
c_val = -4
age_param = -0.0064
age_param_2 = 0.102
h_param = 44
sex_param =0
sex_neigh = 0
sex_diff_param = 2
pct_cost_param = 0
disaster_param_2 =0.05 # parameter for disasters
neighbour_param_2 = 0.37 #parameter for neighbours
germany_param_2 = -1
rich_param = -2
czechia_param = 5
europe_param = 0.65
gdp_param_2 = 0
hcpi_param = 0
students_param = -0.1
dict_group_2 = {'Low income':-0.61, 'Lower middle income':-0.23,
              'Upper middle income':-0.15, 'High income':0.04}
dict_quarter_2 = {1 : -0.1, 2 : 0.5, 3 : -0.1, 4 : 0.5} # dictionary for quarters boosts

cols = ['country', 'year', 'quarter', 'total affected', 'pct_cost', 'delta_gdp',
        'neighbour_dummy', 'gdp_per_capita', 'group', 'pct_students', 'hcpi']

## load simulated population
df = pd.read_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")

# Reshape the dataframe
result = df.melt(id_vars=["agent_id", "country", "sex"], var_name="year", value_name="age")
result["year"] = result["year"].astype(int)
result = result.groupby(["country", "year", "sex"])["age"].count().reset_index()
result.columns = ["country", "year", "sex", "count"]
result = pd.pivot_table(result, index=["country", "year"], columns='sex', values='count')
result['sex_diff'] = abs(result['male'] - result['female'])/(result['male'] + result['female'])
result.drop(columns = ['male', 'female'], inplace = True)
result.reset_index(inplace = True)

df = df[fixed_vars + [str(x) for x in range(2010, 2026)]]
df.columns = fixed_vars + [str(x) for x in range(2010, 2026)]
df = pd.melt(df, id_vars=fixed_vars, value_vars=df.columns[3:],
             value_name='age', var_name='year')
df['year'] = df['year'].astype(int)
# df_all = df.loc[df.index.repeat(4)].reset_index(drop=True)
# quarters = np.tile([1, 2, 3, 4], len(df_all) // 4)
# df_all['quarter'] = quarters
# df = df.merge(result, on = ['country', 'year'], how = 'left')
# df.fillna(0, inplace = True)

## load remittances info
df_rem_quarter = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df_rem_quarter = df_rem_quarter[(df_rem_quarter.group != 0) & (df_rem_quarter.country.isin(df.country.unique().tolist()))]
for year in tqdm(df_rem_quarter.year.unique()):
    df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances'] = (df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())
df_rem_quarter['exp_population'] = df_rem_quarter['remittances'] / 450
df_rem_quarter['probability'] = df_rem_quarter['exp_population'] / df_rem_quarter['population']
df_rem_quarter['probability'] = df_rem_quarter['probability'].clip(0,1)

####
## only big senders??
# df_rem_quarter = df_rem_quarter[df_rem_quarter.remittances > 10_000]
####

##merge everything for the total simulation
# df_all = pd.merge(df_all, df_rem_quarter[cols],
#                              on=['country', 'year', 'quarter'], how='right')
# df_all = df_all[df_all.year > 2012]
# df_all = df_all.merge(result, on = ['country', 'year'], how = 'left')
# df_all.fillna(0, inplace = True)
# df_all.to_pickle("c:\\data\\population\\austria\\population_quarterly_merged.pkl")
df_all = pd.read_pickle("c:\\data\\population\\austria\\population_quarterly_merged.pkl")
df_pred = df_all[(df_all.year > 2020)]
df_all = df_all[(df_all.year > 2012) & (df_all.year < 2020)]

# Optimized probability calculation
def calculate_probability_exponential(
        age, sex, quarters, total_affected,
        neighbour_dummy, country_is_germany, country_is_czechia,
        gdp_per_capita, group_values, students, hcpi, cost,
        europe_list, sex_diff, country_is_rich
):
    # Convert categorical variables to numerical
    male_val = np.where(sex == 'male', 1, 0)
    germany_val = np.where(country_is_germany, 1, 0)
    czechia_val = np.where(country_is_czechia, 1, 0)
    europe_val = np.where(europe_list,1,0)
    female_neigh = np.where((sex == 'female') & (neighbour_dummy == 1), 1, 0)

    # Lookup group values from dict_group_2
    group_vals = np.array([dict_group_2[g] for g in group_values])

    # Calculate the exponent
    exponent = (
            c_val +
            age_param * np.power(age - h_param, 2) +
            age_param_2 * age +
            sex_param * male_val +
            sex_neigh * female_neigh +
            sex_diff_param * sex_diff +
            disaster_param_2 * total_affected +
            neighbour_param_2 * neighbour_dummy +
            quarters +
            germany_param_2 * germany_val +
            czechia_param * czechia_val +
            europe_param * europe_val +
            gdp_param_2 * gdp_per_capita +
            group_vals +
            students * students_param +
            cost * pct_cost_param +
            hcpi * hcpi_param +
            country_is_rich * rich_param
    )

    # Compute the probability
    base_prob = np.exp(exponent) / (1 + np.exp(exponent))
    # Ensure probabilities are within [0, 1]
    base_prob = np.clip(base_prob, 0, 1)
    return base_prob

def simulate_decisions_one_period(year, quarter):
    df_period = df[(df.year == year)].copy().dropna()
    df_period['quarter'] = quarter
    df_period = pd.merge(df_period, df_rem_quarter[cols],
                         on=['country', 'year', 'quarter'], how='left')
    df_period.dropna(inplace=True)

    # Extract necessary arrays
    age = df_period['age'].values
    sex = df_period['sex'].values
    total_affected = df_period['total affected'].values
    neighbour_dummy = df_period['neighbour_dummy'].values
    country_is_germany = df_period['country'].isin(['Germany', 'Switzerland'])
    country_is_czechia = df_period['country'] == 'Czechia'
    country_is_rich = df_period['delta_gdp'] > 0
    europe_list = df_period['country'].isin(europe)
    gdp_per_capita = df_period['gdp_per_capita'].values
    students = df_period['pct_students'].values
    group_values = df_period['group'].values
    quarters = [dict_quarter_2[quarter]] * len(df_period)
    cost = df_period['pct_cost'].values
    hcpi = df_period['hcpi']
    sex_diff = df_period['sex_diff']

    # Calculate probabilities
    probability = calculate_probability_exponential(
        age, sex, quarters, total_affected,
        neighbour_dummy, country_is_germany, country_is_czechia,
        gdp_per_capita, group_values, students,
        hcpi, cost, europe_list, sex_diff, country_is_rich
    )

    # Simulate decisions using numpy
    decision = np.random.binomial(1, probability)

    # Assign back to DataFrame
    df_period['probability'] = probability
    df_period['decision'] = decision

    # Calculate totals
    totals = df_period.groupby('country')['decision'].sum().reset_index()

    # Generate random amounts sent (assuming mean 450)
    # Note: np.random.normal(450, 0, size) will all be 450, consider using a different distribution
    amounts_sent = np.random.normal(amount_sent, deviation_money, len(totals))  # Adjust std as needed
    totals['sim_remittances'] = totals['decision'] * amounts_sent

    # Merge with observed remittances
    observed = df_rem_quarter.loc[
        (df_rem_quarter.year == year) & (df_rem_quarter.quarter == quarter),
        ['country', 'remittances', 'population']
    ].copy()
    totals = pd.merge(totals, observed, on='country', how='left')

    # Calculate error
    totals['error'] = np.abs(totals['remittances'] - totals['sim_remittances'])

    # Rename columns
    totals.rename(columns={'remittances': 'obs_remittances'}, inplace=True)

    # Calculate needed_amount_sent, handle division by zero
    totals['needed_amount_sent'] = np.where(
        totals['decision'] > 0,
        totals['obs_remittances'] / totals['decision'],
        0  # or some appropriate value when decision is zero
    )

    return df_period, totals

def simulate_all_decisions(df_all=df_all):
    # Extract necessary arrays
    age = df_all['age'].values
    sex = df_all['sex'].values
    total_affected = df_all['total affected'].values
    neighbour_dummy = df_all['neighbour_dummy'].values
    country_is_germany = df_all['country'] == 'Germany'
    country_is_czechia = df_all['country'] == 'Czechia'
    europe_list = df_all['country'].isin(europe)
    gdp_per_capita = df_all['gdp_per_capita'].values
    group_values = df_all['group'].values
    students = df_all['pct_students'].values
    quarters = df_all['quarter'].map(dict_quarter_2).values
    cost = df_all['pct_cost'].values
    hcpi = df_all['hcpi']
    sex_diff = df_all['sex_diff']
    country_is_rich = df_all['delta_gdp'] > 0

    # Calculate probabilities
    probability = calculate_probability_exponential(
        age, sex, quarters, total_affected,
        neighbour_dummy, country_is_germany, country_is_czechia,
        gdp_per_capita, group_values, students,
        hcpi, cost, europe_list, sex_diff, country_is_rich
    )

    # Simulate decisions using numpy
    np.nan_to_num(probability, copy = False)
    decision = np.random.binomial(1, probability)

    # Assign back to DataFrame
    df_all['probability'] = probability
    df_all['decision'] = decision

    # Calculate totals
    totals = df_all.groupby(['country', 'year', 'quarter'])['decision'].sum().reset_index()

    # Generate random amounts sent (assuming mean 450)
    # Note: np.random.normal(450, 0, size) will all be 450, consider using a different distribution
    amounts_sent = np.random.normal(amount_sent, deviation_money, len(totals))  # Adjust std as needed
    totals['sim_remittances'] = totals['decision'] * amounts_sent

    # Merge with observed remittances
    observed = df_rem_quarter[['country', 'remittances', 'population', 'year', 'quarter']].copy()
    totals = pd.merge(totals, observed, on=['country', 'year', 'quarter'], how='left')

    # Calculate error
    totals['error'] = np.abs(totals['remittances'] - totals['sim_remittances'])

    # Rename columns
    totals.rename(columns={'remittances': 'obs_remittances'}, inplace=True)

    # Calculate needed_amount_sent, handle division by zero
    totals['needed_amount_sent'] = np.where(
        totals['decision'] > 0,
        totals['obs_remittances'] / totals['decision'],
        0  # or some appropriate value when decision is zero
    )

    return totals

####
# training results
totals = simulate_all_decisions()

totals.dropna(inplace = True)
goodness_of_fit_results(totals, pred=False)

totals['quarter'] = totals['quarter'].astype(str)
totals['year'] = totals['year'].astype(str)
totals['relative_error'] = 100 * totals['error'] / totals['obs_remittances']
fig = px.scatter(totals, x = 'obs_remittances', y='sim_remittances',
                 color = 'country', log_x = True, log_y = True)
fig.add_scatter(x = np.linspace(0,25_000_000, 100),
              y = np.linspace(0,25_000_000, 100))
fig.show()

#prediction results
totals = simulate_all_decisions(df_pred)

totals.dropna(inplace = True)
goodness_of_fit_results(totals, pred=True)

totals['quarter'] = totals['quarter'].astype(str)
totals['year'] = totals['year'].astype(str)
totals['relative_error'] = 100 * totals['error'] / totals['obs_remittances']
fig = px.scatter(totals, x = 'obs_remittances', y='sim_remittances',
                 color = 'country', log_x = True, log_y = True)
fig.add_scatter(x = np.linspace(0,25_000_000, 100),
              y = np.linspace(0,25_000_000, 100))
fig.show()

########################
# one quarter simulation, check probability of sending across whole population
#######################
def percentage_sending_country(country, df = df_all):
    df_country = df[df.country == country].copy()
    df_country = df_country.groupby('year')['probability'].mean()
    df_country.plot(kind = 'bar')
    plt.grid()
    plt.title(f'Percentage population from {country} sending money')
    plt.show(block = True)

percentage_sending_country('Czechia')

def probability_distribution_quarter_year(year, quarter):
    df_period, totals = simulate_decisions_one_period(year, quarter)
    df_period = df_period[['country', 'probability', 'year', 'quarter']].groupby(['country', 'year', 'quarter']).mean()
    df_period.sort_values('probability', inplace=True)
    # df_period.reset_index(drop = True, inplace=True)
    df_period = df_period.merge(df_rem_quarter[(df_rem_quarter.year == year) &
                                   (df_rem_quarter.quarter == quarter)]
                    [['country', 'year', 'quarter', 'probability']], on = ['country', 'year', 'quarter'],
                    how = 'left')
    df_period.sort_values('probability_y', inplace = True)
    df_period.reset_index(inplace = True, drop = True)
    #probability distribution
    fig, ax = plt.subplots()
    ax.plot(df_period['probability_y'], label = 'real values')
    ax.plot(df_period['probability_x'], label = 'fitted values')
    plt.grid()
    plt.legend()
    plt.title('Population probability distribution profile')
    plt.show(block = True)
    #probability distribution
    fig, ax = plt.subplots()
    ax.plot(df_period['probability_y'], label = 'real values')
    df_period.sort_values('probability_x', inplace = True)
    df_period.reset_index(inplace = True, drop = True)
    ax.plot(df_period['probability_x'], label = 'fitted values')
    plt.grid()
    plt.legend()
    plt.title('Population probability distribution profile')
    plt.show(block = True)

probability_distribution_quarter_year(2019, 2)

def plot_needed_incomes_for_perfect_dist(year, quarter):
    df_period, totals = simulate_decisions_one_period(year, quarter)

    #amounts distribution
    fig, ax = plt.subplots()
    ax.hist(totals[totals.needed_amount_sent < 10_000]['needed_amount_sent'], bins = 40)
    plt.grid()
    plt.title('Distribution of perfect theoretical amounts to send')
    plt.show(block = True)
    #amounts scatter
    fig = px.scatter(totals, x='obs_remittances', y='needed_amount_sent',
                     color='country', log_x=True, log_y=False)
    fig.add_hline(450)
    fig.show()

plot_needed_incomes_for_perfect_dist(2022, 2)

####
# test the effects of individual parameters
param_values = [-0.15, -0.1, -0.05, 0]
all_results = pd.DataFrame()
for param in tqdm(param_values):
    dict_quarter_2 = {1 : param, 2 : 0.5, 3 : param, 4 : 0.5}
    for quarter in [1,3]:
        _, totals = simulate_decisions_one_period(2019, quarter)
        totals['year'] = 2019
        totals['quarter'] = str(3)
        totals['param_value'] = str(param)
        r_square = 1 - sum((totals.obs_remittances - totals.sim_remittances) ** 2) / sum(
            (totals.obs_remittances - totals.obs_remittances.mean()) ** 2)
        print(f"r square for {param}: {round(r_square, 3)}")
        all_results = pd.concat([all_results, totals])
all_results['pct_population_sending'] = 100 * all_results['decision'] / all_results['population']
totals = all_results.copy()
totals['pct_population_sending'] = 100 * totals['decision'] / totals['population']

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data = totals, x = 'obs_remittances', y = 'sim_remittances', hue='param_value')
ax.plot(np.linspace(0, max(totals.obs_remittances), 100),
        np.linspace(0, max(totals.obs_remittances), 100), color='red')
plt.xlabel('observed remittances')
plt.ylabel('simulated remittances')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.title('Observed v. simulated remittances in log scale')
plt.show(block=True)

###########
# one year simulations, plot results
###########
def run_sims(year):
    all_results = pd.DataFrame()
    for quarter in tqdm([1,2,3,4]):
        df_period, totals = simulate_decisions_one_period(year, quarter)
        totals['year'] = year
        totals['quarter'] = str(quarter)
        all_results = pd.concat([all_results, totals])
    all_results['pct_population_sending'] = 100 * all_results['decision'] / all_results['population']
    totals = all_results.copy()
    totals['pct_population_sending'] = 100 * totals['decision'] / totals['population']

    plot_all_results_log(totals)

    mean_error = totals[['error', 'obs_remittances']].mean()
    mean_error['relative_error'] = 100 * mean_error['error'] / mean_error['obs_remittances']
    print(mean_error['relative_error'].item())

    r_square = 1 - sum((totals.obs_remittances - totals.sim_remittances)**2)/sum((totals.obs_remittances - totals.obs_remittances.mean())**2)
    print(r_square)

    fig = px.scatter(all_results, x = 'obs_remittances', y='sim_remittances',
                     color = 'country', log_x = True, log_y = True)
    fig.add_scatter(x = np.linspace(0,50_000_000, 100),
                  y = np.linspace(0,50_000_000, 100))
    fig.show()

############################
# optimisation
from scipy.optimize import minimize

# Define the parameters to calibrate
params = ['age_param', 'age_param_2', 'sex_param',
          'disaster_param_2', 'neighbour_param_2',
          'germany_param_2', 'gdp_param_2']

# Initial guess for parameters
initial_params = {
    'age_param': -0.001,
    'age_param_2': 0.1,
    'sex_param': 0.008,
    'disaster_param_2': 0.0042,
    'neighbour_param_2': 0.34,
    'germany_param_2': -0.05,
    'gdp_param_2': -0.96e-05,
}
# Objective function to minimize
def objective_function(x):
    # Map parameters to their names
    global initial_params
    params_dict = dict(zip(params, x))

    # Update global variables or pass them to the function
    # Here, for simplicity, we'll update the global variables
    global age_param, age_param_2, sex_param, disaster_param_2, neighbour_param_2, germany_param_2, gdp_param_2
    age_param = params_dict['age_param']
    age_param_2 = params_dict['age_param_2']
    sex_param = params_dict['sex_param']
    disaster_param_2 = params_dict['disaster_param_2']
    neighbour_param_2 = params_dict['neighbour_param_2']
    germany_param_2 = params_dict['germany_param_2']
    gdp_param_2 = params_dict['gdp_param_2']

    # Simulate decisions for all periods and calculate total error
    total_error = 0
    for year in [2019]:
        for quarter in tqdm(df_rem_quarter.quarter.unique()):
            _, totals = simulate_decisions_one_period(year, quarter)
            total_error += totals['error'].sum()

    return total_error


# Perform optimization
result = minimize(objective_function, x0=list(initial_params.values()),
                  method='L-BFGS-B', bounds=[(-1, 1) for _ in params], options = {'maxiter':100})

# Update parameters with optimized values
optimized_params = dict(zip(params, result.x))
print("Optimized Parameters:", optimized_params)

#plot optimised params results
age_param = optimized_params['age_param']
age_param_2 = optimized_params['age_param_2']
sex_param = optimized_params['sex_param']
disaster_param_2 = optimized_params['disaster_param_2']
neighbour_param_2 = optimized_params['neighbour_param_2']
germany_param_2 = optimized_params['germany_param_2']
gdp_param_2 = optimized_params['gdp_param_2']

all_results = pd.DataFrame()
for quarter in tqdm([1,2,3,4]):
    df_period, totals = simulate_decisions_one_period(2019, quarter)
    totals['year'] = 2019
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

fig = px.scatter(all_results[all_results.country == 'Italy'], x = 'obs_remittances', y='sim_remittances',
                 color = 'quarter', log_x = True, log_y = True)
fig.add_scatter(x = np.linspace(0,50_000_000, 100),
              y = np.linspace(0,50_000_000, 100))
fig.show()
##############################

# iteration over parameter space
# Define parameter ranges
n = 3
age_param_range = np.linspace(-0.004, -0.08, n)
age_param_2_range = np.linspace(0.05, 0.1, n)
h_param_range = np.linspace(40, 50, 2)
sex_param_range = np.linspace(-0.5, 1, n)
disaster_param_range = (0, 0.1, n)
students_param_range = (-1.5, 0, n)
gdp_param_2_range = np.linspace(-1, -10, n) * 1e-05
low_inc_range = np.linspace(-0.2, 0, n)
lm_inc_range = np.linspace(-0.15, 0, n)
um_inc_range = np.linspace(-0.1, 0, n)

# Create the parameter space
parameter_space = list(itertools.product(
    # age_param_range,
    # age_param_2_range,
    # h_param_range,
    sex_param_range,
    disaster_param_range,
    students_param_range,
    gdp_param_2_range,
    low_inc_range,
    lm_inc_range,
    um_inc_range
))

params_df = pd.DataFrame(parameter_space, columns = ['sex',
                                                     'disaster', 'students', 'gdp',
                                                     'low_inc', 'lm_inc', 'um_inc'])

mean_errors = []
for params in tqdm(parameter_space):
    # (age_param, age_param_2, h_param,
    (sex_param, disaster_param_2, students_param, gdp_param_2,
     low_inc, lm_inc, um_inc) = params
    dict_group_2 = {'Low income': low_inc, 'Lower middle income': lm_inc,
                    'Upper middle income': um_inc, 'High income': 0}
    # Simulate decisions for the specified year and quarter
    df_period, totals = simulate_decisions_one_period(2019, 3)

    # Calculate mean error
    mean_error = totals[['error', 'obs_remittances']].mean()
    mean_error['relative_error'] = 100 * mean_error['error'] / mean_error['obs_remittances']
    mean_errors.append(mean_error['relative_error'])

params_df['error'] = mean_errors
params_df.sort_values('error', inplace = True)

# Plot the errors
plt.plot(mean_errors)
plt.title('Relative Error Across Parameter Combinations')
plt.xlabel('Parameter Combination Index')
plt.ylabel('Relative Error (%)')
plt.grid(True)
plt.show(block = True)
###########################
## best parameters combi
age_param, age_param_2, h_param = params_df.iloc[0, :-1]
all_results = pd.DataFrame()
for year in tqdm([2013, 2017, 2021, 2023]):
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
plt.title('Relative error each year')
plt.show(block = True)

fig = px.scatter(all_results, x = 'obs_remittances', y='sim_remittances',
                 color = 'country', log_x = True, log_y = True)
fig.add_scatter(x = np.linspace(0,50_000_000, 100),
              y = np.linspace(0,50_000_000, 100))
fig.show()
