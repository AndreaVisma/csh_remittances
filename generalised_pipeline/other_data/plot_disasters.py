

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time

dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))


def compute_disasters_scores(df, dict_dis_par):
    df_dis = df.copy()
    df_dis = df_dis.drop_duplicates(subset=["date", "origin"])
    for col in ['eq', 'dr', 'fl', 'st']:
        for shift in [int(x) for x in np.linspace(1, 12, 12)]:
            g = df_dis.groupby('origin', group_keys=False)
            g = g.apply(lambda x: x.set_index(['date', 'origin'])[col]
                        .shift(shift).reset_index(drop=True)).fillna(0)
            df_dis[f'{col}_{shift}'] = g.iloc[0]
            df_dis['tot'] = df_dis['fl'] + df_dis['eq'] + df_dis['st'] + df_dis['dr']
    for shift in tqdm([int(x) for x in np.linspace(1, 12, 12)]):
        df_dis[f'tot_{shift}'] = df_dis[f'fl_{shift}'] + df_dis[f'eq_{shift}'] + df_dis[f'st_{shift}'] + df_dis[
            f'dr_{shift}']
    df_dis.rename(columns={'eq': 'eq_0', 'st': 'st_0', 'fl': 'fl_0', 'dr': 'dr_0', 'tot': 'tot_0'}, inplace=True)
    required_columns = ['date', 'origin'] + \
                       [f"{disaster}_{i}" for disaster in ['eq', 'dr', 'fl', 'st', 'tot']
                        for i in range(13)]
    missing_cols = [col for col in required_columns if col not in df_dis.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for disaster in ['eq', 'dr', 'fl', 'st', 'tot']:
        params = dict_dis_par.get(disaster)
        if not params or len(params) != 13:
            raise ValueError(f"Need exactly 13 parameters for {disaster}")
        disaster_cols = [f"{disaster}_{i}" for i in range(13)]
        weights = np.array([params[i] for i in range(13)])
        impacts = df_dis[disaster_cols].values.dot(weights)
        df_dis[f"{disaster}_score"] = impacts
    return df_dis

## pair of countries
origin, destination = "Mexico", "Japan"
diasporas_file = "C:\\Data\\migration\\bilateral_stocks\\complete_stock_hosts_interpolated.pkl"
df = pd.read_pickle(diasporas_file)
df_country = df.query(f"""`origin` == '{origin}' and  `destination` == '{destination}'""")
df_country = df_country[~df_country['date'].duplicated()][['date', 'origin']]

emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters.pkl").rename(columns = {'country' : 'origin'})

dis_params = pd.read_excel("C:\\Data\\my_datasets\\disasters\\disasters_params.xlsx")
dict_dis_par = dict(zip(['eq', 'dr', 'fl', 'st', 'tot'], [dis_params[x].tolist() for x in ['eq', 'dr', 'fl', 'st', 'tot']]))
emdat_country = emdat.query(f"""`origin` == '{origin}'""")
df_country = df_country.merge(emdat_country, on = ['origin', 'date'], how = 'left').fillna(0)
df_country = compute_disasters_scores(df_country, dict_dis_par)

disaster_cols_0 = ['eq_0', 'dr_0', 'fl_0', 'st_0']
score_cols = ['eq_score', 'dr_score', 'fl_score', 'st_score']
disaster_types = ['Earthquake', 'Drought', 'Flood', 'Storm']
dict_dis = dict(zip(disaster_types, disaster_cols_0))
dict_dis_score = dict(zip(disaster_types, score_cols))

# Make sure date is sorted
df_country['date'] = pd.to_datetime(df_country.date)
df_country = df_country.sort_values('date')
df_country = df_country[disaster_cols_0 + score_cols + ["date"]].groupby('date').mean().reset_index()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))

# Panel 1: Bar plot for affected population percentages
df_plot_0 = df_country[['date'] + disaster_cols_0].set_index('date')
if not df_plot_0.empty:
    df_plot_0.plot(kind='bar', stacked=True, ax=axes[0], width=0.8)
    axes[0].set_title("Percentage of Population Affected by Disasters")
    axes[0].set_ylabel("% Affected")
    axes[0].legend(title="Disaster Type")
    axes[0].get_xaxis().set_ticklabels([])
    axes[0].grid(True)
    # Ensure y-axis shows even with small values
    if df_plot_0.sum(axis=1).max() < 0.1:
        axes[0].set_ylim(0, 0.1)
else:
    axes[0].text(0.5, 0.5, 'No data available', ha='center', va='center')
    axes[0].set_title("Percentage of Population Affected by Disasters (No Data)")

# Panel 2: Line plot for disaster scores
df_plot_score = df_country[['date'] + score_cols].set_index('date')
df_plot_score.plot(ax=axes[1])
axes[1].set_title("Disaster Severity Scores Over Time")
axes[1].set_ylabel("Score")
axes[1].legend(title="Disaster Type")
axes[1].grid(True)

# Beautify and show
plt.tight_layout()
plt.show(block=True)

######################################
######################################

# Create figure with 4 panels
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True)
axes = axes.flatten()
for i, disaster_name in enumerate(disaster_types):
    # Create twin axes for each panel (bars on left, line on right)
    ax1 = axes[i]
    ax2 = ax1.twinx()

    # Plot percentage as bars on primary axis
    pct_col = dict_dis[disaster_name]
    df_country[pct_col].plot(kind='bar', ax=ax1, color='skyblue', width=0.8, position=0, label='% Affected')
    ax1.set_ylabel('% Affected', color='skyblue')
    ax1.tick_params(axis='y', colors='skyblue')
    ax1.set_title(f"{disaster_name:} Impact and Severity")

    # Plot score as line on secondary axis
    score_col = dict_dis_score[disaster_name]
    df_country[score_col].plot(ax=ax2, color='crimson', marker='o', label='Severity Score')
    ax2.set_ylabel('Score', color='crimson')
    ax2.tick_params(axis='y', colors='crimson')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Add grid
    ax1.grid(True, axis='y', alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.show(block = True)
