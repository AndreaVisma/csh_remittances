
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import os
from utils import dict_names
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick

# define global variables
data_folder = os.getcwd() + "\\data_downloads\\data\\"

######
# Load all the data in
######

#load inflow of remittances
df_in = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')
df_in = df_in[df_in.year == 2023]
df_in['inflow'] *= 1_000_000

#load outflow of remittances
df_out = pd.read_excel(data_folder + "outward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y", skiprows=2)
df_out.rename(columns = {"Remittance outflows (US$ million)": "country"}, inplace=True)
df_out = pd.melt(df_out, id_vars=['country'], value_vars=df_out.columns.tolist()[1:])
df_out.rename(columns = {"variable": "year", "value" : "outflow"}, inplace=True)
df_out.replace({"2023e": '2023'}, inplace =True)
df_out.year = df_out.year.astype('int')
df_out = df_out[df_out.year == 2023]
df_out['outflow'] *= 1_000_000

##### merge
df_rem = df_in.merge(df_out, on = ['country', 'year'])
print(set(df_rem.country) - set(dict_names.keys()))
df_rem['country'] = df_rem.country.map(dict_names)

#### population
df_pop = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_pop = df_pop[df_pop.year == 2023]
df_rem = df_rem.merge(df_pop, on = ['country', 'year'])

######## GDP
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_clean.xlsx")
df_gdp = df_gdp[df_gdp.year == 2023]
print(set(df_gdp.country) - set(dict_names.keys()))
df_gdp['country'] = df_gdp.country.map(dict_names)
df_rem = df_rem.merge(df_gdp, on = ['country', 'year'])

##### calculate ratios
df_rem['in_per_capita'] = df_rem['inflow'] / df_rem['population']
df_rem['out_per_capita'] = df_rem['outflow'] / df_rem['population']
df_rem['gdp_per_capita'] = df_rem['gdp'] / df_rem['population']

df_rem['pct_in'] = 100 * df_rem['in_per_capita'] / df_rem['gdp_per_capita']
df_rem['pct_out'] = 100 * df_rem['out_per_capita'] / df_rem['gdp_per_capita']
df_rem = df_rem[df_rem.gdp>1e10]

### regional classification
income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)

df_rem = df_rem.merge(income_class[['country', 'group', 'Region']], on = 'country', how = 'left')

##################
## chart with only grey dots
sns.set_theme(style="white")
max_val_size = df_rem.population.max() / df_rem.population.min()
df_rem_small = df_rem[(df_rem.pct_in < 10) & (df_rem.pct_out < 2)]

fig, ax = plt.subplots(figsize = (12,9))
sns.scatterplot(x="pct_out", y="pct_in", size="population",
            sizes=(100, max_val_size), alpha=.8, hue = 'Region',
                data=df_rem_small, ax = ax, palette=["grey"], legend=False)
# Add 1:1 reference line
max_val_x = max(df_rem_small[['pct_out']].max().values)
max_val_y = max(df_rem_small[['pct_in']].max().values)
max_val = min([max_val_x, max_val_y])
ax.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.8)
# plt.legend(False)
plt.show(block = True)
# fig.savefig('.\plots\\for_paper\\multiplier_effect_scatter_grey_inner.png')

df_rem_big = df_rem[(df_rem.pct_in >= 10) | (df_rem.pct_out >= 2)]

fig, ax = plt.subplots(figsize = (12,9))
sns.scatterplot(x="pct_out", y="pct_in", size="population",
            sizes=(100, max_val_size), alpha=.8, hue = 'Region',
                data=df_rem_big, ax = ax, palette=["grey"], legend = False)
# Add 1:1 reference line
max_val_x = max(df_rem_big[['pct_out']].max().values)
max_val_y = max(df_rem_big[['pct_in']].max().values)
max_val = min([max_val_x, max_val_y])
ax.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.8)

ax.set_xlim([-5, max_val_x + 0.5])
ax.set_ylim([-15, max_val_y + 0.5])
# ax.get_legend().remove()
# plt.legend(False)
plt.show(block = True)
fig.savefig('.\plots\\for_paper\\multiplier_effect_scatter_grey_outer.png')

#################

## Diverging Bar Chart
# Aggregate data by income group
fig, ax = plt.subplots(figsize=(8, 5.5))

pastel_colors = ['grey', 'grey', 'grey', 'grey']
groups = df_rem['group'].unique()
group_color_map = {group: pastel_colors[i % len(pastel_colors)] for i, group in enumerate(groups)}

for group in groups:
    subset = df_rem[df_rem['group'] == group]
    plt.scatter(
        subset['pct_out'],
        subset['pct_in'],
        color=group_color_map[group],
        label=group,
        s=100,
        alpha = 0.65
    )

# Add 1:1 reference line
max_val_x = max(df_rem[['pct_out']].max().values)
max_val_y = max(df_rem[['pct_in']].max().values)
max_val = min([max_val_x, max_val_y])
plt.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.65)

# plt.legend()
plt.grid(True)
plt.show(block = True)
fig.savefig('.\plots\\for_paper\\multiplier_effect_scatter_GREY.png', bbox_inches = 'tight')

#####
fig, ax = plt.subplots(figsize=(8, 5.5))

pastel_colors = ['red', 'blue', 'green', 'yellow']
groups = ['Low income', 'Upper middle income', 'Lower middle income',
       'High income']
group_color_map = {group: pastel_colors[i % len(pastel_colors)] for i, group in enumerate(groups)}

for group in groups:
    if group != 'Lower middle income':
        alpha_val = 0
    else:
        alpha_val = 1
    subset = df_rem[df_rem['group'] == group]
    plt.scatter(
        subset['pct_out'],
        subset['pct_in'],
        color=group_color_map[group],
        label=group,
        s=100,
        alpha = alpha_val
    )

max_val_x = max(df_rem[['pct_out']].max().values)
max_val_y = max(df_rem[['pct_in']].max().values)
max_val = min([max_val_x, max_val_y])
plt.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.8)

# plt.legend()
plt.grid(False)
plt.show(block = True)
fig.savefig('.\plots\\for_paper\\multiplier_effect_scatter_LOWMID.png', transparent=True, bbox_inches = 'tight')


fig, ax = plt.subplots(figsize=(8, 5.5))

pastel_colors = ['red', 'blue', 'green', 'red']
groups = ['Low income', 'Upper middle income', 'Lower middle income',
       'High income']
group_color_map = {group: pastel_colors[i % len(pastel_colors)] for i, group in enumerate(groups)}

for group in groups:
    if group != 'High income':
        alpha_val = 0
    else:
        alpha_val = 1
    subset = df_rem[df_rem['group'] == group]
    plt.scatter(
        subset['pct_out'],
        subset['pct_in'],
        color=group_color_map[group],
        label=group,
        s=100,
        alpha = alpha_val
    )

max_val_x = max(df_rem[['pct_out']].max().values)
max_val_y = max(df_rem[['pct_in']].max().values)
max_val = min([max_val_x, max_val_y])
plt.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.8)

# plt.legend()
plt.grid(False)
plt.show(block = True)
fig.savefig('.\plots\\for_paper\\multiplier_effect_scatter_HIGH.png', transparent=True, bbox_inches = 'tight')

### grouped by income
df_group = df_rem[['group', 'pct_in', 'pct_out']].groupby('group').mean()

desired_order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
df_group.index = pd.CategoricalIndex(df_group.index, categories=desired_order, ordered=True)
df_group = df_group.sort_index()

# Bar width and positions
bar_width = 0.35
groups = df_group.index.tolist()
x = np.arange(len(groups))  # positions for groups

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(x - bar_width/2, df_group['pct_in'], width=bar_width, label='inflow of remittances over gdp per capita', color='skyblue')
plt.bar(x + bar_width/2, df_group['pct_out'], width=bar_width, label='outflow of remittances over gdp per capita', color='lightcoral')

# Labeling
plt.xticks(ticks=x, labels=groups, rotation=45, ha='right')
# plt.ylabel('Percentage')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
fig.savefig('.\plots\\for_paper\\multiplier_effect_bars_by_income.svg')
plt.show(block = True)

### grouped by region
df_group = df_rem[['Region', 'pct_in', 'pct_out']].groupby('Region').mean().sort_values('pct_out')

# Bar width and positions
bar_width = 0.35
groups = df_group.index.tolist()
x = np.arange(len(groups))  # positions for groups

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(x - bar_width/2, df_group['pct_in'], width=bar_width, label='inflow of remittances over gdp per capita', color='skyblue')
plt.bar(x + bar_width/2, df_group['pct_out'], width=bar_width, label='outflow of remittances over gdp per capita', color='lightcoral')

# Labeling
plt.xticks(ticks=x, labels=groups, rotation=45, ha='right')
# plt.ylabel('Percentage')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
fig.savefig('.\plots\\for_paper\\multiplier_effect_bars_by_region.svg')
plt.show(block = True)