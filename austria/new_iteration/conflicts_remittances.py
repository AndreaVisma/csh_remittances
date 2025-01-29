
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from utils import dict_names

file = "C:\\Data\\conflict\\GEDEvent_v24_1.csv"

df = pd.read_csv(file, low_memory = False)
df = df[["country", "date_start", "date_end", "best"]]
df.rename(columns = {"best" : "deaths"}, inplace = True)
df['date_start'] = pd.to_datetime(df['date_start'])
df['start_week'] = df['date_start'].apply(lambda x: x - pd.Timedelta(days=x.weekday()))
df['date_end'] = pd.to_datetime(df['date_end'])
df['end_week'] = df['date_end'].apply(lambda x: x - pd.Timedelta(days=x.weekday()))
# df = df[df.date_start >= "01-2012"]
df["country"] = df["country"].map(dict_names)
df["duration"] = (df["date_end"] - df["date_start"]).dt.days + 1
def crosses_week_boundary(start, end):
    # Get the start of the week (Monday) for both start and end dates
    start_week_start = start - pd.Timedelta(days=start.weekday())  # Monday of the start week
    end_week_start = end - pd.Timedelta(days=end.weekday())  # Monday of the end week

    # If the start of the week is different, the event crosses a boundary
    return 1 if start_week_start != end_week_start else 0

df['crosses_weeks'] = df.apply(lambda row: crosses_week_boundary(row['date_start'], row['date_end']), axis=1)
print(f"The events which cross a week boundary are {round(100 * df['crosses_weeks'].sum() / len(df), 2)}% of all events")
print(df.isna().sum())

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
weekly_deaths = (expanded_df.groupby(['country', 'start_week'])['deaths']
                 .sum().reset_index())

def plot_country_conflict_history(country):
    df_country = weekly_deaths[weekly_deaths.country == country]
    min_week, max_week = df_country['start_week'].min(), df_country['start_week'].max()
    all_weeks = pd.date_range(start=min_week, end=max_week, freq='W-MON')  # Weekly frequency, starting on Monday
    complete_df = pd.DataFrame({'start_week': all_weeks})
    merged_df = pd.merge(complete_df, df_country, on='start_week', how='left')
    merged_df['deaths'] = merged_df['deaths'].fillna(0)

    df_country = merged_df[['start_week', 'deaths']].groupby('start_week').sum().reset_index()

    smoothed = savgol_filter(df_country['deaths'], 10, 5)  # window size 13, polynomial order 5

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(df_country['start_week'], df_country['deaths'], color='blue', label='Deaths')
    ax[0].set_title('Weekly Deaths')
    ax[0].grid()
    ax[1].plot(df_country['start_week'], smoothed, color='orange', label='Smoothed Deaths')
    ax[1].set_title('Smoothed Weekly Deaths')
    ax[1].grid()
    fig.suptitle(f"Weekly deaths in conflict in {country}")
    fig.savefig(f"C:\\git-projects\\csh_remittances\\austria\\new_iteration\\plots\\conflict_histories\\{country}.svg")
    plt.show(block=True)

plot_country_conflict_history('Mexico')
plot_country_conflict_history('Syria')
plot_country_conflict_history('South Africa')
plot_country_conflict_history('El Salvador')
plot_country_conflict_history('Brazil')
plot_country_conflict_history('Congo, Dem. Rep.')
plot_country_conflict_history('Ukraine')
plot_country_conflict_history('Israel')
plot_country_conflict_history('Libya')
plot_country_conflict_history('Bosnia')

weekly_deaths.to_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_conflicts.csv", index = False)