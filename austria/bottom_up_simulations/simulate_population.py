"""
Script: simulate_population.py
Author: Andrea Vismara
Date: 12/11/2024
Description: simulate all the diaspora populations over the years
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

## import quarterly data
df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
# df_low = df[df['group'] == 'Low income'].copy()

## import age and sex data
df_dem = pd.read_excel("c:\\data\\population\\austria\\age_sex_all_clean.xlsx")
df_dem = df_dem[df_dem.country.isin(df.country.unique())].sort_values(['country', 'year'])

age_bins = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, 111]  # Define the bins as you see in the target
age_labels = ['[0, 4]','[5, 9]','[10, 14]','[15, 19]','[20, 24]','[25, 29]','[30, 34]','[35, 39]','[40, 44]','[45, 49]','[50, 54]','[55, 59]','[60, 64]',
              '[65, 69]','[70, 74]','[75, 79]','[80, 84]','[85, 89]','[90, 94]','[95, 99]','[100 - 111]']

## function to simulate agents from a row
def simulate_agents(row):
    n = row['people']
    mean_age = row['mean_age']
    if n > 0:
        try:
            age_min, age_max = map(int, row['age_group'][1:-1].split(', '))
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
    for ind, row in df_dem[df_dem.year == years[0]].iterrows():
        new_ages, new_nat, new_sex = simulate_agents(row)
        all_ages = np.concatenate((all_ages, new_ages))
        all_nat = np.concatenate((all_nat, new_nat))
        all_sex = np.concatenate((all_sex, new_sex))
    agents_df = pd.DataFrame({'age': all_ages, 'country': all_nat, 'sex': all_sex})
    agents_df.age = agents_df.age.astype('int')
    agents_df.rename(columns={'age' : str(years[0])}, inplace = True)
    agents_df = agents_df.reset_index().rename(columns={'index': 'agent_id'})

    ## make population age
    agents_df[str(years[0] + 1)] = agents_df[str(years[0])] + 1
    agents_df[f'age_group'] = pd.cut(agents_df[str(years[0] + 1)], bins=age_bins, labels=age_labels, right=True)
    aggregate_next_year = agents_df.groupby(['sex', 'age_group', 'country']).agg(
        people=(str(years[0] + 1), 'size'),  # Count the number of people in each group
    ).reset_index()
    agents_df.drop(columns='age_group', inplace=True)
    agents_df.reset_index(drop = True, inplace = True)

    ## subsequent_years
    for year in tqdm(years[1:]):
        aggregate_next_year = aggregate_next_year.merge(df_dem[df_dem.year == year], on = ['sex', 'age_group', 'country'])
        aggregate_next_year['people'] = aggregate_next_year['people_y'] - aggregate_next_year['people_x']
        aggregate_next_year.drop(columns = ['people_x', 'people_y'], inplace = True)

        ## first remove people
        for _, row in aggregate_next_year[aggregate_next_year.people < 0].iterrows():
            # Define filters based on sex, age_group, and country
            sex = row['sex']
            country = row['country']
            age_group_label = row['age_group']
            people_to_remove = abs(row['people'])

            # Parse the age group label safely
            try:
                # Split the age group label and ensure it has two elements
                age_range = list(map(int, age_group_label.strip("[]").split(", ")))
                if len(age_range) != 2:
                    raise ValueError(f"Invalid age group format: {age_group_label}")
            except Exception as e:
                print(f"Error parsing age group '{age_group_label}': {e}")
                continue  # Skip this row if there's an error

            # Filter the main dataframe for matching records
            mask = (
                    (agents_df['sex'] == sex) &
                    (agents_df['country'] == country) &
                    (agents_df[str(year)].between(age_range[0], age_range[1]))
            )
            df_group = agents_df[mask]

            # If there are enough records to remove
            if len(df_group) >= people_to_remove:
                # Sample random indices to drop
                indices_to_drop = df_group.sample(n=people_to_remove).agent_id
                agents_df.loc[agents_df.agent_id.isin(indices_to_drop), str(year)] = np.nan
            else:
                # Drop all records if not enough to meet the negative count
                agents_df.loc[agents_df.agent_id.isin(df_group.agent_id.unique()), str(year)] = np.nan

        #then add people that were missing
        all_ages, all_nat, all_sex = np.array([]), np.array([]), np.array([])
        for ind, row in aggregate_next_year[aggregate_next_year.people > 0].iterrows():
            new_ages, new_nat, new_sex = simulate_agents(row)
            all_ages = np.concatenate((all_ages, new_ages))
            all_nat = np.concatenate((all_nat, new_nat))
            all_sex = np.concatenate((all_sex, new_sex))

        additional_people = pd.DataFrame({'age': all_ages, 'country': all_nat, 'sex': all_sex})
        additional_people.age = additional_people.age.astype('int')
        additional_people.rename(columns={'age': str(year)}, inplace=True)

        ## concat to all agents
        agents_df.drop(columns = 'agent_id', inplace = True)
        agents_df = pd.concat([agents_df, additional_people]).reset_index(drop = True)
        agents_df = agents_df.reset_index().rename(columns={'index': 'agent_id'})

        ## make population age
        agents_df[str(year + 1)] = agents_df[str(year)] + 1
        agents_df[f'age_group'] = pd.cut(agents_df[str(year + 1)], bins=age_bins, labels=age_labels, right=True)
        aggregate_next_year = agents_df.groupby(['sex', 'age_group', 'country']).agg(
            people=(str(year + 1), 'size'),  # Count the number of people in each group
        ).reset_index()
        agents_df.drop(columns='age_group', inplace=True)
        agents_df.reset_index(drop=True, inplace=True)

    return agents_df

agents_df = simulate_populations(df_dem)

agents_df.to_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")