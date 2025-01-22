
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from utils import dict_names

file = "C:\\Data\\conflict\\GEDEvent_v24_1.csv"

df = pd.read_csv(file)
df = df[["country", "date_start", "date_end", "best"]]
df.rename(columns = {"best" : "deaths"}, inplace = True)
df['date_start'] = pd.to_datetime(df['date_start'])
df['start_week'] = df['date_start'] - df['date_start'].dt.weekday * np.timedelta64(1, 'D')
df['date_end'] = pd.to_datetime(df['date_end'])
df['end_week'] = df['date_end'] - df['date_end'].dt.weekday * np.timedelta64(1, 'D')
df = df[df.date_start >= "01-2012"]
df["country"] = df["country"].map(dict_names)
df.isna().sum()

# Function to expand events into weekly intervals and distribute deaths
def spread_deaths(df):
    expanded_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Generate weekly intervals between start and end dates
        date_range = pd.date_range(start=row['start_week'], end=row['end_week'], freq='W')
        num_weeks = len(date_range)
        if num_weeks == 0:
            # If the event lasts less than a week, assign it to the start date's week
            expanded_rows.append({
                'country': row['country'],
                'start_week': row['start_week'],
                'deaths': row['deaths']
            })
        else:
            # Distribute deaths evenly across weeks
            deaths_per_week = row['deaths'] / num_weeks
            for week_start in date_range:
                expanded_rows.append({
                    'country': row['country'],
                    'start_week': week_start,
                    'deaths': deaths_per_week
                })
    return pd.DataFrame(expanded_rows)

# Expand the DataFrame
expanded_df = spread_deaths(df)

# Group by country and week, summing the deaths
weekly_deaths = (expanded_df.groupby(['country', pd.Grouper(key='start_week', freq='W-MON')])['deaths']
                 .sum().reset_index())

def plot_country_conflict_history(country):
    df_country = weekly_deaths[weekly_deaths.country == country]
    smoothed = savgol_filter(df_country['deaths'], 19, 5)  # window size 13, polynomial order 5

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(df_country['start_week'], df_country['deaths'], color='blue', label='Deaths')
    ax[0].set_title('Weekly Deaths')
    ax[0].grid()
    ax[1].plot(df_country['start_week'], smoothed, color='orange', label='Smoothed Deaths')
    ax[1].set_title('Smoothed Weekly Deaths')
    ax[1].grid()
    fig.suptitle(f"Weekly deaths in conflict in {country}")
    plt.show(block=True)

plot_country_conflict_history('Mexico')

weekly_deaths.to_csv("C:\\Data\\my_datasets\\weekly_conflicts.csv", index = False)