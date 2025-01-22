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
pio.renderers.default = 'browser'
from utils import dict_names

#read the EM-DAT data
emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat["Start Year"] >= 2010) &
              (emdat["Disaster Group"] == "Natural")].copy()
emdat["Country"] = emdat["Country"].map(dict_names)

#create a dataset with dummy variables
df_nd = pd.pivot_table(data=emdat, columns =["Disaster Type"],
                               index=["Country", "Start Year", "Start Month", "Start Day",
                                      "End Year", "End Month", "End Day"], values="Total Affected",
                           aggfunc="sum")
df_nd.fillna(0, inplace = True)
df_nd = df_nd.astype(int)
df_nd["total affected"] = df_nd.sum(axis=1)
df_nd.reset_index(inplace = True)

#clean dates
df_nd[['Start Year','Start Month', 'Start Day']] = (
    df_nd[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
df_nd[['End Year','End Month', 'End Day']] = (
    df_nd[['End Year','End Month', 'End Day']].fillna(1).astype(int))
df_nd["start_date"] = pd.to_datetime(df_nd[['Start Year','Start Month', 'Start Day']].rename(columns=dict(zip(['Start Year','Start Month', 'Start Day'],
                                                                                                              ['year', 'month', 'day']))))
df_nd["end_date"] = pd.to_datetime(df_nd[['End Year','End Month', 'End Day']].rename(columns=dict(zip(['End Year','End Month', 'End Day'],
                                                                                                              ['year', 'month', 'day']))))
df_nd.drop(columns = ['End Year','End Month', 'End Day', 'Start Year','Start Month', 'Start Day'], inplace = True)
df_nd.rename(columns = {'Country' : 'country', "total affected" : "total_affected"}, inplace = True)

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
weekly_disasters = expanded_df.groupby(['country', 'week_start']).sum().reset_index()
weekly_disasters['country'] = weekly_disasters.country.map(dict_names)

weekly_disasters.to_csv("C:\\Data\\my_datasets\\weekly_disasters.csv", index = False)