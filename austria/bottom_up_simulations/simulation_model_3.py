"""
Script: simulation_model.py
Author: Andrea Vismara
Date: 12/11/2024
Description: build a bottom up approach to studying remittances. Third attempt
"""

import pandas as pd
import numpy as np
from austria.bottom_up_simulations.plots.plot_results import *

## global values
alpha = 0.2
beta = 0.05

## import quarterly data
df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df_low = df[df['group'] == 'Low income']

## import age data
df_age = pd.read_excel("C:\\Data\\population\\austria\\age_nationality_hist.xlsx")
df_age = df_age[df_age.country.isin(df_low.country.unique())]

## population at year 0
def simulate_ages(row):
    n = row['people']
    mean_age = row['mean_age']
    if n > 0:
        ages = np.random.normal(loc=mean_age, scale=2, size=n).astype(int)
        age_min, age_max = map(int, row['age_group'][1:-1].split(', '))
        ages = np.clip(ages, age_min, age_max)
        nat = [row['country']] * n
        return ages, nat
    else:
        return np.array([]), np.array([])
def generate_base_population(df):
    agents_df = pd.DataFrame()
    for ind, row in df.iterrows():
        all_ages, all_nat = simulate_ages(row)
        row_df = pd.DataFrame({'age': all_ages, 'country' : all_nat})
        agents_df = pd.concat([agents_df, row_df])

    agents_df.age = agents_df.age.astype('int')
    return agents_df

base_pop = generate_base_population(df_age[df_age.year == 2011])

## agent decision
def agent_decision_age(age, alpha, beta):
        probability = (alpha * age ** 2 * np.e ** (-beta*age) - 25) / 100
        if probability > 0:
            return np.random.binomial(1, probability,1)[0]
        else:
            return 0

base_pop['decision'] = base_pop['age'].apply(lambda x: agent_decision_age(x, alpha=alpha, beta=beta))

totals = base_pop[['country', 'decision']].groupby('country').sum().reset_index()
amounts_sent = dict(zip(totals.country, np.random.normal(300, 50, len(totals))))

def add_error_to_totals(totals, amounts_sent, period):
    year = int(period[:4])
    quarter = int(period[-1:])
    for country in totals.country.unique():
        totals.loc[totals.country == country, 'sim_remittances'] = totals[totals.country == country].decision.item() * amounts_sent[country]
    totals = totals.merge(df_low.loc[(df_low.year == year) & (df_low.quarter == quarter),['country', 'remittances']], on = 'country')
    totals['error'] = abs(totals.remittances - totals.sim_remittances)
    totals['period'] = period
    totals.rename(columns={'remittances': 'obs_remittances'}, inplace=True)
    return totals

totals = add_error_to_totals(totals, amounts_sent, '2011Q1')

def calibration_function(df_age, alpha, beta):
    base_pop = generate_base_population(df_age)
    base_pop['decision'] = base_pop['age'].apply(lambda x: agent_decision_age(x, alpha=alpha, beta=beta))
    totals = base_pop[['country', 'decision']].groupby('country').sum().reset_index()
    amounts_sent = dict(zip(totals.country, np.random.normal(300, 50, len(totals))))

plot_all_results(totals)


