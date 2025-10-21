
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

######################################
# Map disaster-remittance with people affected
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
disasters = ['Drought', 'Earthquake', 'Flood', 'Storm']
disasters_short = ['dr', 'eq', 'fl', 'st']
disaster_names = dict(zip(disasters, disasters_short))

# Use monthly_disasters we created earlier
df_nat = monthly_disasters.copy()

# Keep only relevant disaster types
df_nat = df_nat[df_nat['type'].isin(disasters)].copy()
df_nat = df_nat[(df_nat.month_start.dt.year >= 2010) & (df_nat.month_start.dt.year <=2020)]

cols = ["month_start", "total_affected"]

def group_DIS_by_quarter(df, col_name):
    return (df[cols]
            .groupby(pd.Grouper(key='month_start', freq='Q'))
            .sum()
            .reset_index()
            .rename(columns={'total_affected': col_name}))


floods_aff = group_DIS_by_quarter(df_nat[df_nat.type == "Flood"], 'floods')
storms_aff  = group_DIS_by_quarter(df_nat[df_nat.type == "Storm"], 'storms')
droughts_aff  = group_DIS_by_quarter(df_nat[df_nat.type == "Drought"], 'droughts')
earthquakes_aff  = group_DIS_by_quarter(df_nat[df_nat.type == "Earthquake"], 'earthquakes')

dfs = [
    floods_aff, storms_aff, droughts_aff,
    earthquakes_aff
]

disasters_all = reduce(lambda left, right: pd.merge(left, right, on=['month_start'], how='outer'), dfs)
disasters_all.month_start = disasters_all.month_start - datetime.timedelta(weeks=12)

colors = {
    "earthquakes": "orange",
    "storms": "green",
    "droughts": "red",
    "floods": "blue"
}
# disasters_all[['floods', 'storms', 'droughts', 'earthquakes']] = disasters_all[['floods', 'storms', 'droughts', 'earthquakes']] / 10
ymax_dis = disasters_all[['floods', 'storms', 'droughts', 'earthquakes']].max().max()

for disaster in df_imp.columns:
    fig, ax1 = plt.subplots(figsize=(3.4, 4.6))

    # --- First line: disaster-induced remittances (already in df_imp) ---
    ax1.plot(
        df_imp.index,
        df_imp[disaster],
        color=colors[disaster],
        linewidth=2,
        label="Remittances"
    )
    ax1.set_ylim(0, ymax + 1e8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # ax1.set_ylabel("Remittances", fontsize=12)

    # --- Second line: people affected, pulled from disasters_all ---
    ax2 = ax1.twinx()  # secondary y-axis
    ax2.plot(
        disasters_all["month_start"].iloc[:-3],
        disasters_all[disaster].iloc[:-3],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="People affected"
    )
    ax2.set_ylim(0, ymax_dis + 1e6)
    # ax2.set_yticks([0, 20* 1e6, 40* 1e6, 60* 1e6, 80* 1e6, 100* 1e6, 120* 1e6] )
    # ax2.set_ylabel("People affected", fontsize=12)

    # --- Style ---
    ax1.set_xlabel("", fontsize=13)
    ax1.tick_params(axis="both", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Optional: combine legends from both axes
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    # --- Save ---
    fig.savefig(f'./plots/for_paper/{disaster}_TIMESERIES.svg', bbox_inches='tight')
    # plt.show(block = True)
    plt.close(fig)




