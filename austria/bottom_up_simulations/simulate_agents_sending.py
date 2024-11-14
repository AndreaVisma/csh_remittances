"""
Script: simulate_agents_sending.py
Author: Andrea Vismara
Date: 12/11/2024
Description: simulate the remittance sending process for diasporas over the years
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from tqdm import tqdm
from austria.bottom_up_simulations.plots.plot_results import *

## globals
fixed_vars = ['agent_id', 'country', 'sex']
# Define parameters for probability calculation
alpha = 0.2        # Minimum age to start growing the probability
beta  = 0.05  # Controls the rate of decay for age
gamma = 0.15   # Boost factor if gender is male

## remittances info
df_rem_quarter = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df_rem_year = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")
df_rem_year['remittances'] = df_rem_year.mln_euros * 1_000_000

## simulated population
df = pd.read_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")
df = df[fixed_vars + [str(x) for x in range(2010, 2026)]]
df.columns = fixed_vars + [str(x) for x in range(2010, 2026)]

# Define a vectorized probability function
def calculate_probability(age, sex):
    base_prob = (alpha * age ** 2 * np.e ** (-beta * age) - 18) / 70
    base_prob = max((base_prob - 0.18) / 0.5, 0)  # Adjusted to avoid negative results
    # Apply male boost if sex is male
    if sex == 'male':
        base_prob = base_prob * (1 + gamma)

    return base_prob

def simulate_decisions_one_year(year):
    df_year = df[fixed_vars + [str(year + 1)]].copy().dropna()
    df_year['probability'] = np.vectorize(calculate_probability)(df_year[str(year + 1)], df_year['sex'])
    df_year['decision'] = df_year['probability'].apply(lambda x: np.random.binomial(1, x, 1)[0])
    totals = df_year[['country', 'decision']].groupby('country').sum().reset_index()
    amounts_sent = dict(zip(totals.country, np.random.normal(500 * 4, 50, len(totals))))
    for country in totals.country.unique():
        totals.loc[totals.country == country, 'sim_remittances'] = totals[totals.country == country].decision.item() * amounts_sent[country]
    totals = totals.merge(
        df_rem_year.loc[(df_rem_year.year == year), ['country', 'remittances']], on='country')
    totals['error'] = abs(totals.remittances - totals.sim_remittances)
    totals.rename(columns={'remittances': 'obs_remittances'}, inplace=True)
    return totals

all_results = pd.DataFrame()
for year in tqdm(df_rem_year.year.unique()):
    totals = simulate_decisions_one_year(year)
    all_results = pd.concat([all_results, totals])

plot_all_results_log(all_results)

##plot to verify
import seaborn as sns

f = sns.lineplot(df_2012.iloc[:100000], x='2012', y='probability', hue = 'sex')
plt.grid()
plt.show(block = True)





# # Parameters for the gamma distribution
# k = 2    # Shape parameter
# theta = 40.0  # Scale parameter
#
# # Generate age values
# age_values = np.linspace(0, 100, 1000)
#
# # Calculate the gamma distribution for each age value
# gamma_values = 30 * gamma.pdf(age_values, a=k, scale=theta)
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(age_values, gamma_values, label=f'Gamma-like Function (k={k}, θ={theta})', color='blue')
# plt.xlabel('Age')
# plt.ylabel('Probability based on age')
# plt.legend()
# plt.grid(True)
# plt.show(block =  True)

