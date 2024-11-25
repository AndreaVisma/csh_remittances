import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import itertools
import matplotlib.pyplot as plt
from austria.bottom_up_simulations.plots.plot_results import *
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

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
age_param = -0.002
age_param_2 = 0.15
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
## only big senders??
df_rem_quarter = df_rem_quarter[df_rem_quarter.remittances > 100_000]
####

# Optimized probability calculation without np.vectorize
def calculate_probability_exponential(
        age, sex, quarter, total_affected,
        neighbour_dummy, country_is_germany,
        gdp_per_capita, group_values
):
    # Convert categorical variables to numerical
    sex_val = np.where(sex == 'male', 1, 0)
    country_val = np.where(country_is_germany, 1, 0)

    # Lookup group values from dict_group_2
    group_vals = np.array([dict_group_2[g] for g in group_values])

    # Calculate the exponent
    exponent = (
            age_param * np.power(age, 2) +
            age_param_2 * age +
            sex_param * sex_val +
            disaster_param_2 * total_affected +
            neighbour_param_2 * neighbour_dummy +
            dict_quarter_2[quarter] +
            germany_param_2 * country_val +
            gdp_param_2 * gdp_per_capita +
            group_vals
    )

    # Compute the probability
    base_prob = 1 / (1 + np.exp(exponent))
    # Ensure probabilities are within [0, 1]
    base_prob = np.clip(base_prob, 0, 1)
    return base_prob


def simulate_decisions_one_period(year, quarter):
    df_period = df[df.year == year].copy().dropna()
    df_period['quarter'] = quarter
    df_period = pd.merge(df_period, df_rem_quarter[cols],
                         on=['country', 'year', 'quarter'], how='left')
    df_period.dropna(inplace=True)

    # Extract necessary arrays
    age = df_period['age'].values
    sex = df_period['sex'].values
    total_affected = df_period['total affected'].values
    neighbour_dummy = df_period['neighbour_dummy'].values
    country_is_germany = df_period['country'] == 'Germany'
    gdp_per_capita = df_period['gdp_per_capita'].values
    group_values = df_period['group'].values

    # Calculate probabilities
    probability = calculate_probability_exponential(
        age, sex, quarter, total_affected,
        neighbour_dummy, country_is_germany,
        gdp_per_capita, group_values
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
    amounts_sent = np.random.normal(450, 100, len(totals))  # Adjust std as needed
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
for quarter in tqdm([1,2,3,4]):
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
n = 4
sex_param_range = np.linspace(0.005, 0.02, n)
disaster_param_2_range = np.linspace(0.001, 0.01, n)
neighbour_param_2_range = np.linspace(0.03, 0.1, n)
gdp_param_2_range = np.linspace(-1, -4, n) * 1e-05

# Create the parameter space
parameter_space = list(itertools.product(
    sex_param_range,
    disaster_param_2_range,
    neighbour_param_2_range,
    gdp_param_2_range
))

params_df = pd.DataFrame(parameter_space, columns = ['sex', 'disasters', 'neigh','gdp'])

mean_errors = []
for params in tqdm(parameter_space):
    sex_param, disaster_param_2, neighbour_param_2, gdp_param_2 = params

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
sex_param, disaster_param_2, neighbour_param_2, gdp_param_2 = params_df.iloc[0, :-1]
sex_param = 0.05
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
