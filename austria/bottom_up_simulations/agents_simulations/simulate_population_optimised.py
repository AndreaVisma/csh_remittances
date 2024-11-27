"""
Script: simulate_population.py
Author: Andrea Vismara
Date: 12/11/2024
Description: simulate all the diaspora populations over the years
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import data
df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df_dem = pd.read_excel("c:\\data\\population\\austria\\age_sex_all_clean.xlsx")
df_dem = df_dem[df_dem.country.isin(df.country.unique())].sort_values(['country', 'year'])

# Define age bins and labels
age_bins = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, 111]
age_labels = ['[0, 4]', '[5, 9]', '[10, 14]', '[15, 19]', '[20, 24]', '[25, 29]', '[30, 34]', '[35, 39]', '[40, 44]',
              '[45, 49]', '[50, 54]', '[55, 59]', '[60, 64]', '[65, 69]', '[70, 74]', '[75, 79]', '[80, 84]',
              '[85, 89]', '[90, 94]', '[95, 99]', '[100 - 110]']

# Precompute age group mappings
df_dem['age_min'] = df_dem['age_group'].str.extract(r'\[([0-9]+)').astype(int)
df_dem['age_max'] = df_dem['age_group'].str.extract(r'([0-9]+)\]').astype(int)


# Function to simulate agents in a vectorized manner
def simulate_agents_vectorized(n, age_min, age_max):
    if n.sum() == 0:
        return pd.DataFrame()
    # Generate ages for each group using NumPy
    ages = []
    countries = []
    sexes = []
    for idx in n.index:
        if n[idx] > 0:
            ages_group = np.random.uniform(low=age_min[idx], high=age_max[idx] + 1, size=n[idx]).astype(int)
            countries_group = np.full(n[idx], df_dem.loc[idx, 'country'], dtype=object)
            sexes_group = np.full(n[idx], df_dem.loc[idx, 'sex'], dtype=object)
            ages.append(ages_group)
            countries.append(countries_group)
            sexes.append(sexes_group)
    if not ages:
        return pd.DataFrame()
    # Concatenate arrays
    ages_concat = np.concatenate(ages)
    countries_concat = np.concatenate(countries)
    sexes_concat = np.concatenate(sexes)
    # Create DataFrame
    agents = pd.DataFrame({
        'age': ages_concat,
        'country': countries_concat,
        'sex': sexes_concat
    })
    return agents


# Function to simulate populations
def simulate_populations_optimized(df_dem):
    years = df_dem.year.astype(int).unique().tolist()
    years.sort()

    # Initialize agents DataFrame for year 0
    initial_year = years[0]
    df_initial = df_dem[df_dem.year == initial_year]
    agents = simulate_agents_vectorized(df_initial['people'], df_initial['age_min'], df_initial['age_max'])
    if agents.empty:
        agents = pd.DataFrame({'age': [], 'country': [], 'sex': []})
    agents['agent_id'] = agents.index
    agents = agents.reset_index(drop=True)
    agents.rename(columns={'age': str(initial_year)}, inplace=True)

    # Initialize a list to hold agent data for all years
    agent_data_list = [agents[['agent_id', str(initial_year), 'country', 'sex']]]

    # Precompute age groups for the next year
    agents[str(initial_year + 1)] = agents[str(initial_year)] + 1
    agents['age_group'] = pd.cut(agents[str(initial_year + 1)], bins=age_bins, labels=age_labels, right=True)

    # Group by sex, age_group, country for next year
    group_cols = ['sex', 'age_group', 'country']
    aggregate_next_year = agents.groupby(group_cols, observed=True).size().reset_index(name='people')

    # Remove age_group column from agents
    agents.drop(columns=['age_group'], inplace=True)

    # Prepare df_dem for merging
    # df_dem.set_index(['year'] + group_cols, inplace=True)

    # Iterate over subsequent years
    for year in tqdm(years[1:], desc="Processing years"):
        # Get current population
        current_agents = agents[['agent_id', str(year - 1), 'country', 'sex']].copy()
        current_agents[str(year)] = current_agents[str(year - 1)] + 1
        current_agents['age_group'] = pd.cut(current_agents[str(year)], bins=age_bins, labels=age_labels, right=True)
        current_agents = current_agents.groupby(['sex', 'age_group', 'country']).agg(
            people=(str(years[0] + 1), 'size'),  # Count the number of people in each group
        ).reset_index()

        # Get target population for this year
        group_cols = ['sex', 'age_group', 'country', 'people']
        target_population = df_dem.loc[df_dem.year == year, group_cols].reset_index()

        # Merge current population with target population
        merged = pd.merge(current_agents, target_population, on=['sex', 'age_group', 'country'], how='left',
                          suffixes=('_current', '_target'))

        # Calculate people to add or remove
        merged['people_diff'] = merged['people_target'] - merged['people_current']

        # Remove excess agents
        remove_mask = merged['people_diff'] < 0
        agents_to_remove = merged.loc[remove_mask, 'agent_id']
        agents.loc[agents['agent_id'].isin(agents_to_remove), str(year)] = np.nan

        # Add new agents
        add_mask = merged['people_diff'] > 0
        agents_to_add = merged.loc[add_mask, ['people_diff', 'sex', 'age_group', 'country']]
        agents_to_add.rename(columns={'people_diff': 'people'}, inplace=True)
        new_agents = simulate_agents_vectorized(agents_to_add['people'],
                                                agents_to_add['age_group'].apply(lambda x: x.left),
                                                agents_to_add['age_group'].apply(lambda x: x.right))
        new_agents[str(year)] = new_agents['age']
        new_agents['agent_id'] = agents.index.max() + 1 + np.arange(len(new_agents))
        agent_data_list.append(new_agents[['agent_id', str(year), 'country', 'sex']])

        # Update agents DataFrame
        agents = pd.concat([agents, new_agents], ignore_index=True)
        agents[str(year)] = agents.get(str(year), np.nan)

        # Prepare for next year
        agents['age_group'] = pd.cut(agents[str(year)] + 1, bins=age_bins, labels=age_labels, right=True)
        aggregate_next_year = agents.groupby(group_cols, observed=True).size().reset_index(name='people')
        agents.drop(columns=['age_group'], inplace=True)

    # Concatenate all agent data
    final_agents = pd.concat(agent_data_list, ignore_index=True)
    final_agents.set_index('agent_id', inplace=True)

    return final_agents


# Run the optimized simulation
agents_df = simulate_populations_optimized(df_dem)

# Save the result
agents_df.to_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024_optimized.pkl")