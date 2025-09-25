

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# dictionary of country names
import json
dict_names = {}
with open('c:\\data\\general\\countries_dict.txt',
          encoding='utf-8') as f:
    data = f.read()
js = json.loads(data)
for k,v in js.items():
    for x in v:
        dict_names[x] = k

df_with = pd.read_parquet("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_with_disasters_CORRECT.parquet")
df_month = df_with[(df_with.date.dt.month == 12) & (df_with.date.dt.year == 2019)]

#######
print(f"Total destination countries: {len(set(df_with.destination.unique()).union(set(df_with.origin.unique())))}")
########

df_avg = df_with[["origin", "destination", "sim_remittances"]].groupby(["origin", "destination"]).mean().reset_index()
df_avg_large = df_avg[df_avg.sim_remittances >= 1e6]
######################
# plot heatmap of the matrix

matrix = df_month.pivot(index="destination", columns="origin", values="sim_remittances")

# Plot heatmap
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(np.log1p(matrix), cmap="plasma", cbar = False)
plt.xticks([])
plt.yticks([])
plt.xlabel("")
plt.ylabel("")
fig.savefig('.\plots\\for_paper\\12_2019_heatmap.svg', bbox_inches = 'tight')
plt.show(block = True)

matrix = df_avg.pivot(index="destination", columns="origin", values="sim_remittances")

# Plot heatmap
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(np.log1p(matrix), cmap="plasma", cbar = False)
plt.xticks([])
plt.yticks([])
plt.xlabel("")
plt.ylabel("")
fig.savefig('.\plots\\for_paper\\average_heatmap.svg', bbox_inches = 'tight')
plt.show(block = True)

###
df_without = pd.read_parquet("C:\\git-projects\\csh_remittances\\general\\results_plots\\all_flows_simulations_without_disasters_CORRECT.parquet")
df_avg_without = df_without[["origin", "destination", "sim_remittances"]].groupby(["origin", "destination"]).mean().reset_index()

df_merged = df_avg.merge(df_avg_without, on = ["origin", "destination"], suffixes = ("_with", "_without"))
df_merged["dis_remittances"] = df_merged["sim_remittances_with"] - df_merged["sim_remittances_without"]

matrix = df_merged.pivot(index="destination", columns="origin", values="dis_remittances")

# Plot heatmap
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(np.log1p(matrix), cmap="plasma", cbar = False)
plt.xticks([])
plt.yticks([])
plt.xlabel("")
plt.ylabel("")
fig.savefig('.\plots\\for_paper\\DISASTERS_heatmap.svg', bbox_inches = 'tight')
plt.show(block = True)