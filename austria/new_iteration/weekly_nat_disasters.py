"""
Script: weekly_nat_disasters.py
Author: Andrea Vismara
Date: 16/01/2025
Description: creates a global dataset of natural disaster events at the weekly level
"""

#imports
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import os
import json
from scipy.signal import savgol_filter, gauss_spline
pio.renderers.default = 'browser'
from utils import dict_names

#read the EM-DAT data
emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat["Start Year"] >= 2010) &
              (emdat["Disaster Group"] == "Natural")].copy()
emdat["Country"] = emdat["Country"].map(dict_names)

#create a dataset with dummy variables
df_nd = emdat[["Country", "Start Year", "Start Month", "Start Day","End Year", "End Month", "End Day",
               "Total Affected", "Total Damage, Adjusted ('000 US$)", "Disaster Type"]].copy()
df_nd = df_nd[~df_nd["Start Month"].isna()]

#clean dates
df_nd[['Start Year','Start Month', 'Start Day']] = (
    df_nd[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
df_nd.loc[df_nd["End Month"].isna(), "End Month"] = df_nd.loc[df_nd["End Month"].isna(), "Start Month"]
df_nd[['End Year','End Month', 'End Day']] = (
    df_nd[['End Year','End Month', 'End Day']].fillna(1).astype(int))
df_nd["start_date"] = pd.to_datetime(df_nd[['Start Year','Start Month', 'Start Day']].rename(columns=dict(zip(['Start Year','Start Month', 'Start Day'],
                                                                                                              ['year', 'month', 'day']))))
df_nd["end_date"] = pd.to_datetime(df_nd[['End Year','End Month', 'End Day']].rename(columns=dict(zip(['End Year','End Month', 'End Day'],
                                                                                                              ['year', 'month', 'day']))))
df_nd.drop(columns = ['End Year','End Month', 'End Day', 'Start Year','Start Month', 'Start Day'], inplace = True)
df_nd.rename(columns = {'Country' : 'country', "Total Affected" : "total_affected", "Total Damage, Adjusted ('000 US$)" : "total_damage"}, inplace = True)
df_nd = df_nd[~df_nd.total_affected.isna()]
df_nd["total_damage"] *= 1000
df_nd["duration"] = (df_nd["end_date"] - df_nd["start_date"]).dt.days + 1

df_nd.to_excel("C:\\Data\\natural_disasters\\disasters_start_end_dates.xlsx", index = False)

#########################
########################
def plot_duration_by_disaster_type(country):
    if country != None:
        df_country = df_nd[df_nd.country == country].copy()
        name = country
    else:
        df_country = df_nd.copy()
        name = 'all countries'
    # Use a boxplot or violinplot to show distributions
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size
    sns.stripplot( x='Disaster Type', y='duration', data=df_country, hue='Disaster Type')
    sns.boxplot(x='Disaster Type', y='duration', data=df_country, hue='Disaster Type')

    # Customize labels/titles
    plt.title(f'Distribution of Disaster Durations by Type in {name}', fontsize=14)
    plt.xlabel('Disaster Type', fontsize=12)
    plt.ylabel('Duration (Days)', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-labels for readability
    plt.tight_layout()  # Prevent overlapping elements
    fig.savefig(f"C:\\git-projects\\csh_remittances\\austria\\new_iteration\\plots\\disasters_duration\\{name}.svg")
    plt.show(block = True)
plot_duration_by_disaster_type(None)
plot_duration_by_disaster_type('Pakistan')

##########################
#########################
def crosses_week_boundary(start, end):
    # Get the start of the week (Monday) for both start and end dates
    start_week_start = start - pd.Timedelta(days=start.weekday())  # Monday of the start week
    end_week_start = end - pd.Timedelta(days=end.weekday())  # Monday of the end week

    # If the start of the week is different, the event crosses a boundary
    return 1 if start_week_start != end_week_start else 0

df_nd['crosses_weeks'] = df_nd.apply(lambda row: crosses_week_boundary(row['start_date'], row['end_date']), axis=1)
print(f"The events which cross a week boundary are {round(100 * df_nd['crosses_weeks'].sum() / len(df_nd), 2)}% of all events")

def spread_disasters(df):
    expanded_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Generate weekly intervals between start and end dates
        date_range = pd.date_range(start=row['start_date'], end=row['end_date'], freq='W')
        num_weeks = len(date_range)
        if num_weeks == 0:
            # If the event lasts less than a week, assign it to the start date's week
            expanded_rows.append({
                'country': row['country'],
                'week_start': row['start_date'] - pd.Timedelta(days=row['start_date'].weekday()),
                'total_affected': row['total_affected'],
                'total_damage': row["total_damage"],
                'type': row["Disaster Type"],
                "crosses_weeks": row['crosses_weeks']
            })
        else:
            # Distribute deaths evenly across weeks
            deaths_per_week = row['total_affected'] / num_weeks
            damage_per_week = row["total_damage"] / num_weeks
            for week_start in date_range:
                expanded_rows.append({
                    'country': row['country'],
                    'week_start': week_start - pd.Timedelta(days=week_start.weekday()),
                    'total_affected': deaths_per_week,
                    'total_damage': damage_per_week,
                    'type': row["Disaster Type"],
                    "crosses_weeks": row['crosses_weeks']
                })
    return pd.DataFrame(expanded_rows)

# Expand the DataFrame
expanded_df = spread_disasters(df_nd)

# Group by country and week, summing the deaths
weekly_disasters = (expanded_df.sort_values(['country', 'week_start']).copy())
weekly_disasters['country'] = weekly_disasters.country.map(dict_names)

def plot_country_disaster_history(country):
    if country != None:
        df_country = weekly_disasters[weekly_disasters.country == country].copy()
        name = country
    else:
        df_country = weekly_disasters.copy()
        name = 'all countries'
    min_week, max_week = df_country['week_start'].min(), df_country['week_start'].max()
    all_weeks = pd.date_range(start=min_week, end=max_week, freq='W-MON')  # Weekly frequency, starting on Monday
    complete_df = pd.DataFrame({'week_start': all_weeks})
    merged_df = pd.merge(complete_df, df_country, on='week_start', how='left')
    merged_df['total_affected'] = merged_df['total_affected'].fillna(0)
    merged_df['total_damage'] = merged_df['total_damage'].fillna(0)

    df_country = merged_df[['week_start', 'total_affected', 'total_damage']].groupby('week_start').sum().reset_index()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(df_country['week_start'], df_country['total_affected'], color='blue', label='Disasters affected people')
    ax[0].set_title('Number of people affected by disasters per week')
    ax[0].grid()
    ax[1].plot(df_country['week_start'], df_country['total_damage'], color='orange', label='Smoothed Disasters affected people')
    ax[1].set_title('Total damages from disasters per week')
    ax[1].grid()
    fig.suptitle(f"Weekly disasters in {name}")
    fig.savefig(f"C:\\git-projects\\csh_remittances\\austria\\new_iteration\\plots\\disasters_histories\\{name}.svg")
    plt.show(block=True)

plot_country_disaster_history('Mexico')
plot_country_disaster_history('China')
plot_country_disaster_history('Bangladesh')
plot_country_disaster_history('Pakistan')
plot_country_disaster_history(None)

weekly_disasters.to_csv("C:\\Data\\my_datasets\\weekly_disasters.csv", index = False)