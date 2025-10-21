
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
emdat = emdat[(emdat["Start Year"] >= 2000) &
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

def spread_disasters_monthly(df):
    expanded_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Generate month starts between start and end
        date_range = pd.date_range(start=row['start_date'], end=row['end_date'], freq='MS')
        num_months = len(date_range)
        if num_months == 0:
            # If the event lasts less than a month, assign it to the start month
            expanded_rows.append({
                'country': row['country'],
                'month_start': row['start_date'].replace(day=1),
                'total_affected': row['total_affected'],
                'total_damage': row["total_damage"],
                'type': row["Disaster Type"]
            })
        else:
            # Distribute values evenly across months
            affected_per_month = row['total_affected'] / num_months
            damage_per_month = row["total_damage"] / num_months
            for month_start in date_range:
                expanded_rows.append({
                    'country': row['country'],
                    'month_start': month_start,
                    'total_affected': affected_per_month,
                    'total_damage': damage_per_month,
                    'type': row["Disaster Type"]
                })
    return pd.DataFrame(expanded_rows)

# Expand disasters to monthly level
monthly_disasters = spread_disasters_monthly(df_nd)

# Relevant disasters
disasters = ['Drought', 'Earthquake', 'Flood', 'Storm']
disasters_short = ['dr', 'eq', 'fl', 'st']
disaster_names = dict(zip(disasters, disasters_short))

# Use monthly_disasters we created earlier
df_nat = monthly_disasters.copy()

# Keep only relevant disaster types
df_nat = df_nat[df_nat['type'].isin(disasters)].copy()

# Extract year for population merge
df_nat['year'] = df_nat['month_start'].dt.year

# Load population data
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")

# Merge with population
df_nat = df_nat.merge(df_pop_country, on=['country', 'year'], how='left')

# Keep both absolute and relative values
df_nat['share_affected'] = 100 * df_nat['total_affected'] / df_nat['population']

# Pivot to wide format by disaster type
df_nat_pivot = df_nat.pivot_table(
    index=['month_start', 'country'],
    columns='type',
    values='share_affected',
    aggfunc='sum'
).reset_index()

# Rename columns to short names
df_nat_pivot.rename(columns=disaster_names, inplace=True)
df_nat_pivot.rename(columns={"month_start" : "date"}, inplace=True)

def ensure_full_monthly_index(df):
    result = []
    for country, group in df.groupby("country"):
        # Build a complete monthly date range for that country
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="MS")
        group = group.set_index("date").reindex(full_range).reset_index()
        group["country"] = country
        group = group.rename(columns={"index": "date"})
        # Fill missing values with 0 (no disasters that month)
        disaster_cols = ["dr", "eq", "fl", "st"]
        group[disaster_cols] = group[disaster_cols].fillna(0)
        result.append(group)
    return pd.concat(result, ignore_index=True)

df_nat_pivot = ensure_full_monthly_index(df_nat_pivot)

# --- now proceed with lag creation as before ---

# Sort before lags
df_nat_pivot = df_nat_pivot.sort_values(["country", "date"]).copy()

# Add lags for each disaster
disaster_cols = ['dr', 'eq', 'fl', 'st']
for col in disaster_cols:
    for lag in range(1, 13):
        df_nat_pivot[f"{col}_{lag}"] = (
            df_nat_pivot.groupby("country")[col].shift(lag)
        )

# Add totals (current + lags)
df_nat_pivot["tot"] = df_nat_pivot[disaster_cols].sum(axis=1)
for lag in range(1, 13):
    lagged_cols = [f"{col}_{lag}" for col in disaster_cols]
    df_nat_pivot[f"tot_{lag}"] = df_nat_pivot[lagged_cols].sum(axis=1)

df_nat_pivot.fillna(0, inplace = True)
df_nat_pivot.rename(columns = {'country' : 'origin', 'eq' : 'eq_0', 'fl' : 'fl_0', 'dr' : 'dr_0', 'st' : 'st_0'}, inplace=True)
df_nat_pivot.date = df_nat_pivot.date + pd.offsets.MonthEnd(0)
# Save
df_nat_pivot.to_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")

