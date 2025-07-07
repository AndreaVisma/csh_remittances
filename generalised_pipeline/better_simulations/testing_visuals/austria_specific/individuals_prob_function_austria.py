
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
from utils import zero_values_before_first_positive_and_after_first_negative, dict_names


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

## diaspora info Austria
df_at = pd.read_excel("c:\\data\\migration\\austria\\population_by_nationality_year_land_2010-2024.xlsx")
df_at = df_at.melt(id_vars = 'year', value_vars=df_at.columns[2:],
             value_name='total_n_people', var_name='origin')
df_at.loc[df_at['total_n_people'] == '-', 'total_n_people'] = 0
df_at = df_at.groupby(['year', 'origin']).sum()
df_at = df_at.reset_index()
df_at['origin'] = df_at['origin'].map(dict_names)
df_at = df_at.dropna()
df_at.year = df_at.year.astype(int) - 1
df_at = df_at[df_at.year == 2020]

####
df_f = df.copy()
country = "Austria"
results_dfs = []
df_country = df_f[df_f.destination == country].copy()
df_nta_ita = df_nta.query(f"""`country` == '{country}'""")[['age', 'nta']].fillna(0)
df_country['nta'] = df_country.age_group.str[:2]
df_country['nta'] = df_country['nta'].str.replace('-', '').astype(int) + 1
for ind, row in df_nta_ita.iterrows():
    nta_dict[int(row.age)] = round(row.nta, 2)
def est_theta(row):
    return (param_nta * (nta_dict[row['nta']] - 1))  \
                + (param_asy * row['asymmetry']) + (param_gdp * row['gdp_diff_norm']) + constant
df_country['theta'] = df_country.apply(est_theta, axis = 1)
df_country['probability'] = 1 / (1 + np.exp(-df_country['theta']))
results_dfs.append(df_country)

df_res = pd.concat(results_dfs)
df_res.sort_values('n_people', inplace = True, ascending = False)

def country_summary(group: pd.DataFrame) -> pd.Series:
    total_n = group['n_people'].sum()
    # np.average accepts weights:
    w_avg = np.average(group['probability'], weights=group['n_people'])
    return pd.Series({
        'total_n_people': total_n,
        'weighted_probability': w_avg
    })

country_stats = df_res.groupby('origin').apply(country_summary).reset_index()
country_stats = country_stats.drop(columns = 'total_n_people')
country_stats = country_stats.merge(df_at[['origin', 'total_n_people']], on = 'origin', how = 'left')
country_stats['sums'] = country_stats['weighted_probability'] * country_stats['total_n_people']
tot_avg = country_stats['sums'].sum() / country_stats['total_n_people'].sum()
df_plot = country_stats.copy()
country_stats['total_n_people'] /= 1_000
country_stats['weighted_probability'] *= 100

import seaborn as sns
import matplotlib.ticker as mtick

sns.set_style('white')
max_val_size = 15 * country_stats.total_n_people.max()
fig, ax = plt.subplots(figsize=(12, 9))
plt.scatter(country_stats['total_n_people'], country_stats['weighted_probability'],
            color = 'grey', s = 150, alpha = 0.7)

for _, row in country_stats.iterrows():
    x = row['total_n_people']
    y = row['weighted_probability']
    country = row['origin']
    plt.text(x + 5, y + 0.002, country, fontsize=12)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.grid(True)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# fig.savefig('.\plots\\for_paper\\AUSTRIA_individuals_probability.png', bbox_inches = 'tight')
plt.show(block = True)


#############################

# produce entry and exit view

df_plot.drop(columns='sums', inplace = True)

#### population
df_pop = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_pop = df_pop[df_pop.year == 2020]
df_pop.rename(columns = {'country' : 'origin'}, inplace = True)
df_pop = df_pop[['origin', 'population']]
df_plot = df_plot.merge(df_pop, on = 'origin')

######## GDP
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_clean.xlsx")
df_gdp = df_gdp[df_gdp.year == 2020]
df_gdp['origin'] = df_gdp.country.map(dict_names)
df_gdp = df_gdp[['origin', 'gdp']]
df_plot = df_plot.merge(df_gdp, on = 'origin')

##### calculate ratios
gdp_per_cap_austria = df_gdp[df_gdp.origin == "Austria"].gdp.item() / df_pop[df_pop.origin == "Austria"].population.item()
df_plot['remittances'] = 0.18 * df_plot['total_n_people'] * df_plot['weighted_probability'] * gdp_per_cap_austria

df_plot['gdp_per_capita'] = df_plot['gdp'] / df_plot['population']
df_plot['in_per_capita'] = df_plot['remittances'] / df_plot['population']
df_plot['out_per_capita'] = df_plot['remittances'] / df_pop[df_pop.origin == "Austria"].population.item()

df_plot['pct_in'] = 100 * df_plot['in_per_capita'] / df_plot['gdp_per_capita']
df_plot['pct_out'] = 100 * df_plot['out_per_capita'] / gdp_per_cap_austria

max_val_size = 2 * df_plot.population.max() / df_plot.population.min()
fig, ax = plt.subplots(figsize = (12,9))
sns.scatterplot(x="pct_out", y="pct_in", size="population",
            sizes=(200, max_val_size), alpha=.8, hue = 'origin',
                data=df_plot, ax = ax, palette=["grey"], legend=False)
# Add 1:1 reference line
max_val_x = max(df_plot[['pct_out']].max().values)
max_val_y = max(df_plot[['pct_in']].max().values)
max_val = min([max_val_x, max_val_y])
ax.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.8)
# plt.legend(False)

# for _, row in df_plot.iterrows():
#     x = row['pct_out']
#     y = row['pct_in']
#     country = row['origin']
#     ax.text(x + 0.001, y + 0.001, country, fontsize=10)

plt.grid(True)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("")
plt.ylabel("")
fig.savefig('.\plots\\for_paper\\AUSTRIA_multiplier_effect_scatter_grey_inner.png')
plt.show(block = True)


###########################
# effect of disasters in remittances Austria

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
constant = -2.5

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

def simulate_all_countries_austria(height, shape):
    global nta_dict
    df_sim = df[df.destination == "Austria"].copy()
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

#########
rem_per_period_without = simulate_all_countries_austria(height = 0, shape = 0)

rem_per_period_with= simulate_all_countries_austria(height = 0.215, shape = 0.52)

res = rem_per_period_without.merge(rem_per_period_with, on = ['date', 'origin', 'destination'], suffixes = ('_without', '_with'))
res = res[['date', 'sim_remittances_without', 'sim_remittances_with']].groupby('date').sum().reset_index()
res.set_index('date', inplace = True)
res['difference_bn'] = (res['sim_remittances_with'] - res['sim_remittances_without']) / 1_000_000_000
res['sim_remittances_with'] /= 1_000_000_000
res['sim_remittances_without'] /= 1_000_000_000
res['pct_growth'] = 100 * (res['sim_remittances_with'] - res['sim_remittances_without']) / res['sim_remittances_without']

print(f"Total simulated remittances without: {res['sim_remittances_without'].sum()} billions")
print(f"Total simulated remittances with: {res['sim_remittances_with'].sum()} billions")
print(f"Difference: {1_000 * (res['sim_remittances_with'].sum() - res['sim_remittances_without'].sum())} millions")
print(f"PCT growth = {100 * (res['sim_remittances_with'].sum() - res['sim_remittances_without'].sum())/ res['sim_remittances_without'].sum()}")

################
# compare with KNOMAD

fig, ax = plt.subplots(figsize = (8,8))
res.plot(ax = ax)
plt.legend().remove()
plt.grid(True)
fig.savefig('.\plots\\for_paper\\AUSTRIA_rem_with_without.png')
plt.show(block = True)