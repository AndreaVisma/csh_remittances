
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from functools import reduce
dict_names = {}
with open('c:\\data\\general\\countries_dict.txt',
          encoding='utf-8') as f:
    data = f.read()
js = json.loads(data)
for k,v in js.items():
    for x in v:
        dict_names[x] = k

##################################
# import disasters dataframes
df_without = pd.read_parquet("./general/results_plots/all_flows_simulations_without_disasters_CORRECT.parquet")
df_with = pd.read_parquet("./general/results_plots/all_flows_simulations_with_disasters_CORRECT.parquet")
droughts = pd.read_parquet("./general/results_plots/CORRECT_drought.parquet")
floods = pd.read_parquet("./general/results_plots/CORRECT__floods.parquet")
storms = pd.read_parquet("./general/results_plots/CORRECT__storms.parquet")
earthquakes = pd.read_parquet("./general/results_plots/CORRECT__earthquakes.parquet")

### calculate totals per period
# Calculate totals per quarter
cols = ["date", "sim_remittances"]

# Function to group by quarter
def group_by_quarter(df, col_name):
    return (df[cols]
            .groupby(pd.Grouper(key='date', freq='Q'))
            .sum()
            .reset_index()
            .rename(columns={'sim_remittances': col_name}))

# Group all dataframes by quarter
df_with_quarter = group_by_quarter(df_with, 'with')
df_without_quarter = group_by_quarter(df_without, 'without')
floods_quarter = group_by_quarter(floods, 'floods')
storms_quarter = group_by_quarter(storms, 'storms')
droughts_quarter = group_by_quarter(droughts, 'droughts')
earthquakes_quarter = group_by_quarter(earthquakes, 'earthquakes')

# Merge all dataframes
data_frames = [
    df_with_quarter, df_without_quarter, floods_quarter,
    storms_quarter, droughts_quarter, earthquakes_quarter
]

df_merged = reduce(lambda left, right: pd.merge(left, right, on=['date'], how='outer'), data_frames)

# Calculate differences
df_diff = pd.DataFrame({
    "diff_with_without": df_merged["with"] - df_merged["without"],
    "diff_floods": df_merged["floods"] - df_merged["without"],
    "diff_storms": df_merged["storms"] - df_merged["without"],
    "diff_droughts": df_merged["droughts"] - df_merged["without"],
    "diff_earthquakes": df_merged["earthquakes"] - df_merged["without"]
})

df_diff["tot_indiv"] = (df_diff["diff_floods"] + df_diff["diff_storms"] +
                       df_diff["diff_droughts"] + df_diff["diff_earthquakes"])
df_diff["how_far"] = df_diff["diff_with_without"] - df_diff["tot_indiv"]
## fix first and last value
# df_diff["diff_with_without"].iloc[0] *= 3
df_diff["diff_with_without"].iloc[-1] *= 3

# Plot the difference
df_diff.set_index(df_merged.date)['how_far'].plot()
plt.title("Total all disasters minus sum individual disasters (Quarterly)")
plt.show(block=True)

# Calculate shares
disaster_types = ["floods", "storms", "droughts", "earthquakes"]
for disaster in disaster_types:
    df_diff[f"share_{disaster}"] = df_diff[f"diff_{disaster}"] / df_diff["tot_indiv"]

import datetime
import matplotlib.ticker as mticker

# Calculate impacts
df_imp = pd.DataFrame()
for disaster in disaster_types:
    df_imp[disaster] = df_diff[f"share_{disaster}"] * df_diff["diff_with_without"]
df_imp.index = df_merged.date - datetime.timedelta(weeks=12)

# Plot stacked impacts
fig, ax = plt.subplots(figsize=(6.5, 6.5))
# df_diff.set_index(df_merged.date)['diff_with_without'].plot()

plt.stackplot(
    df_imp.index,
    df_imp["earthquakes"]/1e9,
    df_imp["storms"]/1e9,
    df_imp["droughts"]/1e9,
    df_imp["floods"]/1e9,
    labels=["Earthquakes", "Storms", "Droughts", "Floods"],
    alpha=0.7,
    colors = ['orange', 'green', 'red',  'blue']
)

# plt.legend(loc="upper left")
# plt.title("Temporal Evolution of Natural Disasters (Quarterly, Stacked)")
plt.xlabel("")
plt.ylabel("")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig('.\plots\\for_paper\\indiv_disasters_TIMESERIES.svg', bbox_inches = 'tight')
plt.show(block=True)

###########

# Find the global max across all disasters for consistent y-axis scale
ymax = df_imp.max().max()

# Define colors for each disaster
colors = {
    "earthquakes": "orange",
    "storms": "green",
    "droughts": "red",
    "floods": "blue"
}

# Loop through each disaster type and make filled area plot
for disaster in df_imp.columns:
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.fill_between(
        df_imp.index,
        df_imp[disaster],
        color=colors[disaster],
        alpha=0.7
    )

    # Apply consistent y-axis limit
    ax.set_ylim(0, ymax)

    # Style
    # ax.set_title(f"{disaster.capitalize()} over Time", fontsize=16)
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    # Save as SVG
    fig.savefig(f'./plots/for_paper/{disaster}_TIMESERIES.svg', bbox_inches='tight')
    plt.close(fig)

########################################
### Now try relationship between affected people and remittances
########################################
df_tot = df_imp.sum(axis = 0).reset_index().rename(columns = {"index" : "disaster", 0 : "remittances"})

emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat["Start Year"] >= 2010) & (emdat["Start Year"] < 2020) &
              (emdat["Disaster Type"].isin(["Earthquake", 'Flood', "Storm", "Drought"]))].copy()
emdat["Country"] = emdat["Country"].map(dict_names)

dict_dis = dict(zip(["Earthquake", 'Flood', "Storm", "Drought"], ["earthquakes", 'floods', "storms", "droughts"]))


affected_by_dis = (emdat[["Disaster Type", "Total Affected"]].rename(columns = {"Disaster Type" : "disaster"})
                       .groupby("disaster").sum().sort_values("Total Affected", ascending = False).reset_index())
affected_by_dis["disaster"] = affected_by_dis["disaster"].map(dict_dis)
df_tot = df_tot.merge(affected_by_dis, on = "disaster")

### calculate shares
df_tot["share_rem"] = df_tot["remittances"] / df_tot["remittances"].sum()
df_tot["share_affected"] = df_tot["Total Affected"] / df_tot["Total Affected"].sum()
df_tot["rem_per_affected"] = df_tot["remittances"] / df_tot["Total Affected"]
########## plot it

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mtick


df_tot = df_tot.sort_values('share_affected', ascending=True)

fig, ax = plt.subplots(figsize=(10, 10))
square = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(square)
cumulative_x = 0
cumulative_y = 0

colors = ['blue', 'red', 'green', 'orange']

for i, row in df_tot.iterrows():
    # Create rectangle that starts from cumulative position and extends diagonally
    rect = patches.Rectangle(
        (cumulative_x, cumulative_y),  # bottom left corner (current cumulative position)
        row['share_affected'],  # width (x-direction)
        row['share_rem'],  # height (y-direction)
        fill=True,
        alpha=0.7,
        color=colors[i],
        label=f"{row['disaster']} (rem: {row['share_rem']:.3f}, aff: {row['share_affected']:.3f})"
    )
    ax.add_patch(rect)

    # Update cumulative position for next rectangle
    cumulative_x += row['share_affected']
    cumulative_y += row['share_rem']

# ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect correlation (1:1)')

# ax.set_ylabel('Share of Remittances', fontsize=12)
# ax.set_xlabel('Share of Total Affected', fontsize=12)
# ax.set_title('Stacked Disaster Impact: Remittances vs Affected Population', fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
fig.savefig('.\plots\\for_paper\\square_disasters.svg', bbox_inches = 'tight')
plt.show(block = True)

################
# INVERTED
df_tot = df_tot.sort_values('share_affected', ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
square = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(square)
cumulative_x = 0
cumulative_y = 0

colors = ['blue', 'green', 'red', 'orange']

for i, row in df_tot.iterrows():
    # Create rectangle that starts from cumulative position and extends diagonally
    rect = patches.Rectangle(
        (cumulative_x, cumulative_y),  # bottom left corner (current cumulative position)
        row['share_affected'],  # width (x-direction)
        row['share_rem'],  # height (y-direction)
        fill=True,
        alpha=0.7,
        color=colors[i],
        label=f"{row['disaster']} (rem: {row['share_rem']:.3f}, aff: {row['share_affected']:.3f})"
    )
    ax.add_patch(rect)

    # Update cumulative position for next rectangle
    cumulative_x += row['share_affected']
    cumulative_y += row['share_rem']

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
fig.savefig('.\plots\\for_paper\\square_disasters_INVERTED.svg', bbox_inches = 'tight')
plt.show(block = True)
