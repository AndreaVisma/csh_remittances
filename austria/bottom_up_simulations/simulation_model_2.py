"""
Script: simulation_model.py
Author: Andrea Vismara
Date: 06/11/2024
Description: build a bottom up approach to studying remittances. Second attempt
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize, dual_annealing
import matplotlib.pyplot as plt
from austria.bottom_up_simulations.plots.plot_results import *


### global values
base_prob = 0.012
mean_remittance = 0.0048  # Mean remittance sent in a year in mln EUR
beta_1 = 3.227e-06 # for income
beta_2 = -1.087e-06 # for gdp
beta_3 = -1.256e-03 # for dependency
beta_4 = 1.542e-01 # for neighbour dummy

### import data
df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")
df.rename(columns = {'pop' : 'population', 'mln_euros' : 'remittances'}, inplace = True)

def simulate_remittance_decision_per_country(row, base_prob, beta_1, beta_2, beta_3,
                                             beta_4, mean_remittance):
    # Adjust probability based on income or other factors if needed
    probability = (base_prob + beta_1 * row['income'] + beta_2 * row['gdp']
                   + beta_3 * row["dep_ratio"] + beta_4 * row['neighbour_dummy'])
    if probability < 0:
        probability = 0
    people_sending = row['population'] * probability #expected value of the probability
    amount_sent = mean_remittance #expected value of the normal distribution
    return people_sending * amount_sent


def simulate_all_countries(base_prob, beta_1, beta_2, beta_3,
                           beta_4, mean_remittance):
    # Empty list to hold aggregate remittances per country
    aggregate_remittances = []

    for _, row in df.iterrows():
        # Calculate the total remittance for this country
        total_remittance = simulate_remittance_decision_per_country(row, base_prob, beta_1, beta_2, beta_3,
                                             beta_4, mean_remittance)
        aggregate_remittances.append({
            'country': row['country'],
            'sim_remittances': total_remittance
        })
    results_df = pd.DataFrame(aggregate_remittances)
    results_df['obs_remittances'] = df['remittances']
    results_df['sq_err'] = (results_df['sim_remittances'] - results_df['obs_remittances']) ** 2
    results_df['tot_sum_sq'] = (results_df['obs_remittances'] - results_df['obs_remittances'].mean()) ** 2
    return results_df

# Convert aggregate results to DataFrame
results_df = simulate_all_countries(base_prob, beta_1, beta_2, beta_3,
                           beta_4, mean_remittance)
plot_all_results(results_df)
plot_results_country(results_df, 'Germany')

def calibration_function(params):
    base_prob, beta_1, beta_2, beta_3, beta_4, mean_remittance = params
    result = simulate_all_countries(base_prob, beta_1, beta_2, beta_3, beta_4, mean_remittance)
    result['obs_remittances'] = df['remittances']
    total_error = sum((result.sim_remittances - result.obs_remittances) ** 2)
    return total_error

###try differential evolution method for the optimisation
parameter_bounds = {
    'base_prob': (0.0, 1),
    'beta_1': (-0.1, 0.1),
    'beta_2': (-0.1, 0.1),
    'beta_3': (-0.5, 0.5),
    'beta_4': (-1.0, 1.0),
    'mean_remittance': (0.001, 0.01)
}
bounds = [parameter_bounds[x] for x in parameter_bounds.keys()]
result = dual_annealing(func=calibration_function, bounds=bounds)

print(f"""Optimized parameters:
===============================
base probability: {round(result.x[0], 5)}
income: {round(result.x[1], 5)}
gdp per capita: {round(result.x[2], 5)}
dependency ratio: {round(result.x[3], 5)}
neighbour dummy: {round(result.x[4], 5)}
average remittances in a year: {round(1_000_000 * result.x[5], 5)} EUR""")

opt_param = result.x
results_df = simulate_all_countries(opt_param[0], opt_param[1], opt_param[2], opt_param[3],
                           opt_param[4], opt_param[5])

print(f"R squared: {1 - (results_df['sq_err'].sum() / results_df['tot_sum_sq'].sum())}")

calibration_function(opt_param)