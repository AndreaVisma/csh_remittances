
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from functools import reduce
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
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

df_in = df_with[["origin", "sim_remittances"]].groupby("origin").sum().reset_index()
df_in_ = df_without[["origin", "sim_remittances"]].groupby("origin").sum().reset_index()

df_merge = df_in.merge(df_in_, on = "origin", suffixes = ("_with", "_without"))
df_merge["disaster_rem"] = df_merge["sim_remittances_with"] - df_merge["sim_remittances_without"]
df_merge.loc[df_merge.disaster_rem == 0, "disaster_rem"] = 0.001
df_merge["log_disasters"] = np.log(df_merge["disaster_rem"])

df_in["log_remittances"] = np.log(df_in["sim_remittances"])


################
# EMDAT
emdat = pd.read_excel("c:\\data\\natural_disasters\\emdat_2024_07_all.xlsx")
emdat = emdat[(emdat["Start Year"] >= 2010) & (emdat["Start Year"] < 2020) &
              (emdat["Disaster Type"].isin(["Earthquake", 'Flood', "Storm", "Drought"]))].copy()
emdat["Country"] = emdat["Country"].map(dict_names)

dis_by_country = (emdat[["Country", "Total Affected"]].rename(columns = {"Country" : "origin"})
                       .groupby("origin").sum().sort_values("Total Affected", ascending = False).reset_index())

## population
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_pop_country = df_pop_country[['country', 'population']].groupby('country').mean().reset_index()

set(dis_by_country.origin) - set(df_pop_country.country)
set(df_pop_country.country) - set(dis_by_country.origin)

dis_by_country = df_pop_country.merge(dis_by_country, left_on = 'country', right_on = 'origin', how = "left").fillna(0)
dis_by_country['pct_affected'] = (dis_by_country["Total Affected"] * 10 ) / dis_by_country['population']

##############
## affected as pct of population per country / year

bins_dis = [0, 1, 5, 10, 15, 20, dis_by_country["pct_affected"].max()]
labels_dis = [
    "< 1%", "1-5%", "5-10%", "10-15%", "15-20%", "> 20%"
]
dis_by_country["pct_affected_bins"] = pd.cut(
    dis_by_country["pct_affected"],
    bins=bins_dis,
    labels=labels_dis,
    include_lowest=True
).astype(pd.CategoricalDtype(categories=labels_dis, ordered=True))
dis_by_country.sort_values("pct_affected_bins", inplace = True)
dis_by_country.loc[dis_by_country.origin == "CAR", 'country'] = "Central African Republic"

colors = px.colors.sample_colorscale("Reds", [i/6 for i in range(7)])
color_map = dict(zip(labels_dis, colors))

# Choropleth with bins
fig = px.choropleth(
    dis_by_country,
    locations="country",
    locationmode="country names",
    color="pct_affected_bins",   # categorical bins
    hover_name="country",
    hover_data={"pct_affected": ":,.0f"},  # show absolute values nicely
    color_discrete_map=color_map
)

style = "natural earth"
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type=style
    ),
       width=1800, height=1000
)

fig.write_image(f'.\\plots\\for_paper\\disaster_mao_BINS.svg')
fig.show()

############################
# rem per person per year

df_merge = df_pop_country.merge(df_merge, left_on = 'country', right_on = 'origin', how = "left").fillna(0)
df_merge['disaster_rem_per_person_year'] = (0.1 * df_merge["disaster_rem"]) / df_merge["population"]
df_merge['rem_per_person_year'] = (0.1 * df_merge["sim_remittances_with"]) / df_merge["population"]

bins_rem = [0, 1, 5, 15, 35, 75, 100, df_merge["disaster_rem_per_person_year"].max()]
labels_rem = [
    "< 1", "1 - 5", "5 - 15", "15 - 35", "35 - 75", "75 - 100", "> 100"
]
df_merge["dis_rem_bins"] = pd.cut(
    df_merge["disaster_rem_per_person_year"],
    bins=bins_rem,
    labels=labels_rem,
    include_lowest=True
).astype(pd.CategoricalDtype(categories=labels_rem, ordered=True))
df_merge.sort_values("dis_rem_bins", inplace = True)
df_merge.loc[df_merge.origin == "CAR", 'country'] = "Central African Republic"

colors = px.colors.sample_colorscale("speed", [i/7 for i in range(8)])
color_map = dict(zip(labels_rem, colors))

# Choropleth with bins
fig = px.choropleth(
    df_merge,
    locations="country",
    locationmode="country names",
    color="dis_rem_bins",   # categorical bins
    hover_name="country",
    hover_data={"disaster_rem_per_person_year": ":,.0f"},  # show absolute values nicely
    color_discrete_map=color_map
)

style = "natural earth"
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type=style
    ),
       width=1800, height=1000
)

fig.write_image(f'.\\plots\\for_paper\\DISASTER_rem_per_person_year_BINS.svg')
fig.show()



##############################
df_merge_dis = df_merge.merge(dis_by_country, on = "origin", how = "left")
df_merge_dis["Total Affected"].fillna(1, inplace = True)
df_merge_dis["rem_per_affected"] = df_merge_dis["disaster_rem"] / df_merge_dis["Total Affected"]
df_merge_dis.sort_values("rem_per_affected", ascending = False, inplace = True)

bins = [0, 1e6, 5e7, 5e8, 1e9, 1e10, df_merge["disaster_rem"].max()]
labels = [
    "< 1M", "1–50M", "50–500M", "500M–1B", "1B-10B", "> 10B"
]

# Create ordered categorical
df_merge["disaster_rem_bins"] = pd.cut(
    df_merge["disaster_rem"],
    bins=bins,
    labels=labels,
    include_lowest=True
).astype(pd.CategoricalDtype(categories=labels, ordered=True))
df_merge.sort_values("disaster_rem_bins", inplace = True)
df_merge.loc[df_merge.origin == "CAR", 'origin'] = "Central African Republic"

colors = px.colors.sample_colorscale("speed", [i/6 for i in range(7)])
color_map = dict(zip(labels, colors))

# Choropleth with bins
fig = px.choropleth(
    df_merge,
    locations="origin",
    locationmode="country names",
    color="disaster_rem_bins",   # categorical bins
    hover_name="origin",
    hover_data={"disaster_rem": ":,.0f"},  # show absolute values nicely
    color_discrete_map=color_map
)

style = "natural earth"
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type=style
    ),
       width=1800, height=1000
)

fig.write_image(f'.\\plots\\for_paper\\remittances_map_{style}_BINS.svg')
fig.show()

######################
# disaster remittances

fig = px.choropleth(
    df_merge, #[df_merge.disaster_rem >= 1_000_000],
    locations="origin",
    locationmode="country names",
    color="log_disasters",  # use log scale
    hover_name="origin",
    color_continuous_scale="RdYlGn"
)
# fig.update(layout_coloraxis_showscale=False)
style = "natural earth"
fig.update_layout(
    # title=f"{style}",
    xaxis_title="",
    yaxis_title="",
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type=style)
    # ),
    # colorbar=dict(len=0.75,
    #                   title='#Cases',
    #                   x=0.9,
    #                   tickvals = [-1, 0, 1, 2, 3, 3.699, 4],
    #                   ticktext = ['1', '10', '100', '1000', '5000','10000'])
)
fig.write_image(f'.\plots\\for_paper\\remittances_map_{style}_LEGEND.svg')
fig.show()

############################

fig = px.choropleth(
    df_merge,
    locations="origin",
    locationmode="country names",
    color="log_disasters",  # use log scale
    hover_name="origin",
    color_continuous_scale="Greens"
)
fig.update(layout_coloraxis_showscale=False)
for style in ['airy', 'aitoff', 'albers', 'albers usa', 'august',
            'azimuthal equal area', 'azimuthal equidistant', 'baker',
            'bertin1953', 'boggs', 'bonne', 'bottomley', 'bromley',
            'collignon', 'conic conformal', 'conic equal area', 'conic equidistant', 'craig', 'craster', 'cylindrical equal area', 'cylindrical stereographic', 'eckert1', 'eckert2',
            'eckert3', 'eckert4', 'eckert5', 'eckert6', 'eisenlohr',
            'equal earth', 'equirectangular', 'fahey', 'foucaut',
            'foucaut sinusoidal', 'ginzburg4', 'ginzburg5',
            'ginzburg6', 'ginzburg8', 'ginzburg9', 'gnomonic',
            'gringorten', 'gringorten quincuncial', 'guyou', 'hammer',
            'hill', 'homolosine', 'hufnagel', 'hyperelliptical',
            'kavrayskiy7', 'lagrange', 'larrivee', 'laskowski',
            'loximuthal', 'mercator', 'miller', 'mollweide', 'mt flat polar parabolic', 'mt flat polar quartic', 'mt flat polar sinusoidal', 'natural earth', 'natural earth1', 'natural earth2', 'nell hammer', 'nicolosi', 'orthographic',
            'patterson', 'peirce quincuncial', 'polyconic',
            'rectangular polyconic', 'robinson', 'satellite', 'sinu mollweide', 'sinusoidal', 'stereographic', 'times',
            'transverse mercator', 'van der grinten', 'van der grinten2', 'van der grinten3', 'van der grinten4',
            'wagner4', 'wagner6', 'wiechel', 'winkel tripel',
            'winkel3']:
    fig.update_layout(
       title = f"{style}",
       xaxis_title="",
       yaxis_title="",
       geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type=style
            ),
       # width=600, height=700
    )
    fig.write_image(f'.\plots\\for_paper\\geo_styles\\{style}.svg')
    # fig.show()