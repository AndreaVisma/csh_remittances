

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

df_avg = df_with[["origin", "destination", "sim_remittances"]].groupby(["origin", "destination"]).sum().reset_index()
df_avg['sim_remittances'] /= 10
# df_avg_large = df_avg[df_avg.sim_remittances >= 1e6]
######################
# plot heatmap of the matrix

# matrix = df_month.pivot(index="destination", columns="origin", values="sim_remittances")
#
# # Plot heatmap
# fig, ax = plt.subplots(figsize=(8,8))
# sns.heatmap(np.log1p(matrix), cmap="plasma", cbar = False)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel("")
# plt.ylabel("")
# fig.savefig('.\plots\\for_paper\\12_2019_heatmap.svg', bbox_inches = 'tight')
# plt.show(block = True)

##########
income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)
income_class = income_class[["country", "group"]]
##########


df_avg = df_avg.merge(income_class, left_on = 'origin', right_on = 'country', how = 'left')
df_avg = df_avg.merge(income_class, left_on = 'destination', right_on = 'country', how = 'left',
                      suffixes = ("_or", "_dest"))
df_avg["group_or"] = df_avg.group_or.astype(pd.CategoricalDtype(categories=['High income', 'Upper middle income',
                                                                    'Lower middle income', 'Low income'], ordered=True))
df_avg["group_dest"] = df_avg.group_dest.astype(pd.CategoricalDtype(categories=['High income', 'Upper middle income',
                                                                    'Lower middle income', 'Low income'], ordered=True))

# Sort dataframe
df_avg.sort_values(
    ["group_dest", "group_or", "sim_remittances"],
    ascending=[True, True, False],
    inplace=True
)

df_large = df_avg[df_avg.sim_remittances >= 1e5].copy()
# Pivot into matrix
matrix = df_large.pivot(index="destination", columns="origin", values="sim_remittances")

# --- Reorder rows and columns based on income groups and remittance size ---
# Order destinations
dest_order = (
    df_large.groupby("destination")[["group_dest", "sim_remittances"]]
    .max()  # keep group + largest remittance
    .sort_values(["group_dest", "sim_remittances"], ascending=[True, False])
    .index
)

# Order origins
orig_order = [x for x in dest_order if x in matrix.columns]
# (
#     df_avg.groupby("origin")[["group_or"]]
#     .max()
#     .sort_values(["group_or"], ascending=[True])
#     .index
# )

# Apply ordering to matrix
matrix = matrix.loc[dest_order, orig_order]

from matplotlib.colors import LogNorm, Normalize

# Plot heatmap
fig, ax = plt.subplots(figsize=(6.3,6.3))
sns.heatmap(matrix, cmap="gist_heat_r", square=True, cbar = True, norm=LogNorm())
plt.xticks([])
plt.yticks([])
plt.xlabel("")
plt.ylabel("")
ax.set_facecolor('white')
fig.savefig('.\plots\\for_paper\\average_heatmap_GIST_COLORMAP.svg', bbox_inches = 'tight')
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