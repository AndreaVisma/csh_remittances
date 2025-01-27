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
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import os
import json
from scipy.signal import savgol_filter
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
                'week_start': row['start_date'],
                'total_affected': row['total_affected']
            })
        else:
            # Distribute deaths evenly across weeks
            deaths_per_week = row['total_affected'] / num_weeks
            for week_start in date_range:
                expanded_rows.append({
                    'country': row['country'],
                    'week_start': week_start,
                    'total_affected': deaths_per_week
                })
    return pd.DataFrame(expanded_rows)

# Expand the DataFrame
expanded_df = spread_disasters(df_nd)

# Group by country and week, summing the deaths
weekly_disasters = (expanded_df.groupby(['country', pd.Grouper(key='week_start', freq='W-MON')])['total_affected']
                 .sum().reset_index())
weekly_disasters['country'] = weekly_disasters.country.map(dict_names)

def plot_country_disaster_history(country):
    df_country = weekly_disasters[weekly_disasters.country == country]
    smoothed = savgol_filter(df_country['total_affected'], 19, 5)  # window size 13, polynomial order 5

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(df_country['week_start'], df_country['total_affected'], color='blue', label='Disasters affected people')
    ax[0].set_title('Weekly Affected by disaster')
    ax[0].grid()
    ax[1].plot(df_country['week_start'], smoothed, color='orange', label='Smoothed Disasters affected people')
    ax[1].set_title('Smoothed Weekly Disasters')
    ax[1].grid()
    fig.suptitle(f"Weekly disasters in {country}")
    plt.show(block=True)

plot_country_disaster_history('China')

weekly_disasters.to_csv("C:\\Data\\my_datasets\\weekly_disasters.csv", index = False)