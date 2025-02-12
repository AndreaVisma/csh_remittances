"""
Script: simulate_population.py
Author: Andrea Vismara
Date: 12/02/2024
Description: simulate all the diaspora populations over the years in italy
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

## import quarterly data
df = pd.read_parquet("C:\\Data\\my_datasets\\italy\\simulation_data.parquet")

## import age and sex data
df_dem = pd.read_csv('c:\\data\\migration\\italy\\estimated_stocks_new.csv')
df_dem.rename(columns = {'citizenship' : 'country'}, inplace = True)
df_dem = df_dem[df_dem.country.isin(df.country.unique())].sort_values(['country', 'year'])
df_dem.loc[df_dem.age_group.isin(['Less than 5 years', '100 years or over']), 'age_group'] = (
    df_dem.loc[df_dem.age_group.isin(['Less than 5 years', '100 years or over']), 'age_group'].map(
        dict(zip(['Less than 5 years', '100 years or over'], ['From 0 to 5 years', 'From 100 to 104 years']))
    ))

##### divide by 10!
df_dem['count'] = df_dem['count'] // 10 #each agent will be 10 people
#####
age_bins = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, 104]  # Define the bins as you see in the target
age_labels = ['From 0 to 5 years', 'From 5 to 9 years', 'From 10 to 14 years', 'From 15 to 19 years',
              'From 20 to 24 years', 'From 25 to 29 years','From 30 to 34 years', 'From 35 to 39 years',
              'From 40 to 44 years', 'From 45 to 49 years', 'From 50 to 54 years',
              'From 55 to 59 years', 'From 60 to 64 years', 'From 65 to 69 years','From 70 to 74 years',
              'From 75 to 79 years', 'From 80 to 84 years', 'From 85 to 89 years','From 90 to 94 years',
              'From 95 to 99 years', 'From 100 to 104 years']

## function to simulate agents from a row
def simulate_agents(row):
    n = row['count']
    if n > 0:
        try:
            age_min, age_max = map(int, re.findall(r'\d+', row['age_group']))
        except:
            age_min, age_max = 100, 110
        ages = np.random.uniform(low=age_min, high=age_max+1, size=n).astype(int)
        nat = np.array([row['country']] * n)
        sex = np.array([row['sex']] * n)
        return ages, nat, sex
    else:
        return np.array([]), np.array([]), np.array([])

## function to simulate the population for all the years
def simulate_populations(df_dem):
    years = df_dem.year.unique().tolist()

    ## year 0
    all_ages, all_nat, all_sex = np.array([]), np.array([]), np.array([])
    for ind, row in tqdm(df_dem[df_dem.year == years[0]].iterrows(), total = len(df_dem[df_dem.year == years[0]])):
        new_ages, new_nat, new_sex = simulate_agents(row)
        all_ages = np.concatenate((all_ages, new_ages))
        all_nat = np.concatenate((all_nat, new_nat))
        all_sex = np.concatenate((all_sex, new_sex))
    agents_df = pd.DataFrame({'age': all_ages, 'country': all_nat, 'sex': all_sex})
    agents_df['yrs_since_mig'] = np.rint(np.random.normal(loc=6, scale=2, size=len(agents_df))).astype(int)
    agents_df.loc[agents_df['yrs_since_mig'] < 0, 'yrs_since_mig'] = 0
    agents_df.age = agents_df.age.astype('int')
    agents_df.rename(columns={'age' : f"{years[0]}_age", 'yrs_since_mig' : f"{years[0]}_yrs_since_mig"}, inplace = True)
    agents_df = agents_df.reset_index().rename(columns={'index': 'agent_id'})

    ## make population age
    agents_df[f"{years[0] + 1}_age"] = agents_df[f"{years[0]}_age"] + 1
    agents_df[f"{years[0] + 1}_yrs_since_mig"] = agents_df[f"{years[0]}_yrs_since_mig"] + 1

    agents_df['age_group'] = pd.cut(agents_df[f"{years[0] + 1}_age"], bins=age_bins, labels=age_labels, right=True)
    aggregate_next_year = agents_df.groupby(['sex', 'age_group', 'country']).agg(
        count=(f"{years[0] + 1}_age", 'size'),  # Count the number of people in each group
    ).reset_index()
    agents_df.drop(columns='age_group', inplace=True)
    agents_df.reset_index(drop = True, inplace = True)

    ## subsequent_years
    for year in tqdm(years[1:]):
        aggregate_next_year = aggregate_next_year.merge(df_dem[df_dem.year == year], on = ['sex', 'age_group', 'country'])
        aggregate_next_year['count'] = aggregate_next_year['count_y'] - aggregate_next_year['count_x']
        aggregate_next_year.drop(columns = ['count_x', 'count_y'], inplace = True)

        ## first remove people
        for _, row in aggregate_next_year[aggregate_next_year['count'] < 0].iterrows():
            # Define filters based on sex, age_group, and country
            sex = row['sex']
            country = row['country']
            age_group_label = row['age_group']
            people_to_remove = abs(row['count'])

            # Parse the age group label safely
            try:
                # Split the age group label and ensure it has two elements
                age_range = [int(x) for x in re.findall(r'\d+', age_group_label)]
                if len(age_range) != 2:
                    raise ValueError(f"Invalid age group format: {age_group_label}")
            except Exception as e:
                print(f"Error parsing age group '{age_group_label}': {e}")
                continue  # Skip this row if there's an error

            # Filter the main dataframe for matching records
            mask = (
                    (agents_df['sex'] == sex) &
                    (agents_df['country'] == country) &
                    (agents_df[f"{year}_age"].between(age_range[0], age_range[1]))
            )
            df_group = agents_df[mask]

            # If there are enough records to remove
            if len(df_group) >= people_to_remove:
                # Sample random indices to drop
                indices_to_drop = df_group.sample(n=people_to_remove).agent_id
                agents_df.loc[agents_df.agent_id.isin(indices_to_drop), f"{year}_age"] = np.nan
            else:
                # Drop all records if not enough to meet the negative count
                agents_df.loc[agents_df.agent_id.isin(df_group.agent_id.unique()), f"{year}_age"] = np.nan

        #then add people that were missing
        all_ages, all_nat, all_sex = np.array([]), np.array([]), np.array([])
        for ind, row in aggregate_next_year[aggregate_next_year['count'] > 0].iterrows():
            new_ages, new_nat, new_sex = simulate_agents(row)
            all_ages = np.concatenate((all_ages, new_ages))
            all_nat = np.concatenate((all_nat, new_nat))
            all_sex = np.concatenate((all_sex, new_sex))

        additional_people = pd.DataFrame({f"{year}_age": all_ages, 'country': all_nat, 'sex': all_sex})
        additional_people[f"{year}_yrs_since_mig"] = 0
        additional_people[f"{year}_age"] = additional_people[f"{year}_age"].astype('int')

        ## concat to all agents
        agents_df.drop(columns = 'agent_id', inplace = True)
        agents_df = pd.concat([agents_df, additional_people]).reset_index(drop = True)
        agents_df = agents_df.reset_index().rename(columns={'index': 'agent_id'})

        ## make population age
        agents_df[f"{year + 1}_age"] = agents_df[f"{year}_age"] + 1
        agents_df[f"{year + 1}_yrs_since_mig"] = agents_df[f"{year}_yrs_since_mig"] + 1
        agents_df[f'age_group'] = pd.cut(agents_df[f"{year + 1}_age"], bins=age_bins, labels=age_labels, right=True)
        aggregate_next_year = agents_df.groupby(['sex', 'age_group', 'country']).agg(
            count=(f"{year + 1}_age", 'size'),  # Count the number of people in each group
        ).reset_index()
        agents_df.drop(columns='age_group', inplace=True)
        agents_df.reset_index(drop=True, inplace=True)

    return agents_df

agents_df = simulate_populations(df_dem)
agents_df.set_index(['agent_id', 'country', 'sex'], inplace = True)
agents_df = agents_df.astype('Int64')
agents_df.to_pickle("c:\\data\\population\\italy\\simulated_migrants_populations_2008_2022.pkl")