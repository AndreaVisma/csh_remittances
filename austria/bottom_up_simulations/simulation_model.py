"""
Script: simulation_model.py
Author: Andrea Vismara
Date: 06/11/2024
Description: build a bottom up approach to studying remittances
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize, OptimizeResult

### import data
df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")
df.rename(columns = {'pop' : 'population', 'mln_euros' : 'remittances'}, inplace = True)
df = df[df.year == 2023]

### simulate remittances
# Parameters for the binomial distribution (initial guess)
base_prob = 0.2  # base probability of sending remittances

def simulate_remittance_decision(row, base_prob):
    # Adjust probability based on income or other factors if needed
    probability = base_prob
    return np.random.binomial(1, probability, row['population'])

### simulate remittances amounts
# Parameters for the normal distribution (initial guess)
mean_remittance = 500  # Mean remittance amount in EUR
std_dev_remittance = 100  # Standard deviation of remittance amount

def simulate_remittance_amount(decisions, mean_remittance):
    return np.where(decisions == 1, np.random.normal(mean_remittance, 0, len(decisions)), 0)

def simulate_all_countries():
    # Empty list to hold aggregate remittances per country
    aggregate_remittances = []

    for _, row in tqdm(df.iterrows(), total = len(df)):
        # Simulate decision for each individual in the country
        decisions = simulate_remittance_decision(row, base_prob)

        # Simulate remittance amounts for those who decided to send
        amounts = simulate_remittance_amount(decisions, mean_remittance)

        # Calculate the total remittance for this country
        total_remittance = np.sum(amounts)
        aggregate_remittances.append({
            'country': row['country'],
            'total_remittance': total_remittance
        })
    return pd.DataFrame(aggregate_remittances)

# Convert aggregate results to DataFrame
results_df = simulate_all_countries()
print(results_df)

### optimize parameters
def calibration_function(params):
    base_prob, mean_remittance = params
    total_error = 0

    for _, row in tqdm(df.iterrows(), total = len(df)):
        # Simulate remittance decision and amounts with current parameters
        decisions = simulate_remittance_decision(row, base_prob)
        amounts = simulate_remittance_amount(decisions, mean_remittance)

        # Aggregate simulated remittances
        total_remittance = np.sum(amounts)

        # Compare with observed remittances and calculate error
        observed_remittance = df.loc[(df.country == row['country']) & (df.year == row['year']), 'remittances']
        total_error += (total_remittance - observed_remittance) ** 2  # Sum of squared errors

    return total_error


# Initial parameter guesses
initial_guess = [0.05, 0.0004]

# Optimize parameters to minimize the calibration error
bounds = [(0, 1), (0, 0.001)]
result = minimize(fun = calibration_function,
                  x0 = initial_guess,
                  method='BFGS')

OptimizeResult(result)
# Extract optimized parameters
optimized_base_prob, optimized_mean_remittance = result.x
print("Optimized parameters:", optimized_base_prob, optimized_mean_remittance)
