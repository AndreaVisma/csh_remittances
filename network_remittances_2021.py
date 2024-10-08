"""
Script: remittances_data_download.py
Author: Andrea Vismara
Date: 01/07/2024
Description:
"""

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import plotly.io as pio
pio.renderers.default = "browser"

from utils import *

# define global variables
data_folder = os.getcwd() + "\\data_downloads\\data\\"

df = pd.read_excel(data_folder + "\\bilateral_remittance_matrix_2021.xlsx",
                   skiprows = 1, nrows = 215)
df.rename(columns = {df.columns[0]: "sending_country"}, inplace = True)
df = pd.melt(df, id_vars=['sending_country'], value_vars=df.columns.tolist()[1:])
df.dropna(inplace = True)
df.rename(columns = {"variable" : "receiving_country", "value" : "mln_remittances"}, inplace = True)
df["sending_country"] = clean_country_series(df["sending_country"])
df["receiving_country"] = clean_country_series(df["receiving_country"])

###########
## print some stats
###########
print(f"""Total remittances in 2021: {round(df["mln_remittances"].sum() / 2_000, 2)} bn USD""")

print("================================")
print("Remittances flows for")
for country in df.receiving_country.unique():
    print(f"""{country} || in: {round(df[df.receiving_country == country]["mln_remittances"].sum() / 1_000, 2)} bn USD; out: {round(df[df.sending_country == country]["mln_remittances"].sum() / 1_000, 2)} bn USD""")

##################
world = gpd.read_file("C:\\Data\\geo\\admin_0\\ne_110m_admin_0_countries.shp")[["ADMIN", "geometry"]]
world.geometry = world.geometry.representative_point()
world['lon'] = world.geometry.x
world['lat'] = world.geometry.y
world["ADMIN"] = clean_country_series(world["ADMIN"])

df = df.merge(world[["ADMIN", "lat", "lon"]], left_on="sending_country", right_on="ADMIN", how="inner")
df.rename(columns = {"lat" : "lat_start", "lon" : "lon_start"}, inplace = True)
df.drop(columns = "ADMIN", inplace = True)
df = df.merge(world[["ADMIN", "lat", "lon"]], left_on="receiving_country", right_on="ADMIN", how="inner")
df.rename(columns = {"lat" : "lat_end", "lon" : "lon_end"}, inplace = True)
df.drop(columns = "ADMIN", inplace = True)

cmap = plt.get_cmap('Spectral')

def inflows_country_2021(country):

    df_country = df[(df.receiving_country == country) &
                    (df.mln_remittances > 0.5)].sort_values(
        "mln_remittances", ascending=False
    ).reset_index(drop = True).head(15)
    # df_country.mln_remittances = df_country.mln_remittances.apply(lambda x: np.log(x))

    fig = go.Figure()
    for i in tqdm(range(len(df_country))):
        lons, lats = shortest_path([df_country['lon_start'][i], df_country['lat_start'][i]],
                                   [df_country['lon_end'][i], df_country['lat_end'][i]],
                                   dir = 1, n=100)
        color = mpl.colors.rgb2hex(cmap(df_country["mln_remittances"][i] / df_country["mln_remittances"].max()))
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines+markers',
                line=dict(width=2,
                          color=color),
                marker = dict(showscale=True,
                              size = 0,
                              color = color,
                              colorscale= "spectral", #[[0, mpl.colors.rgb2hex(cmap(0))],
                                          #[1, mpl.colors.rgb2hex(cmap(1))]],
                              cmin=df_country["mln_remittances"].min(),
                              cmax=df_country["mln_remittances"].max()),
                name = df_country["sending_country"][i],
                hovertext= f"mln USD: {round(df_country['mln_remittances'][i], 2)}"

            )
        )
    fig.update_layout(
        margin={'l': 100, 't': 40, 'b': 100, 'r': 100},
        mapbox={
            'center': {'lon': 10, 'lat': 10},
            'style': "open-street-map",
            'center': {'lon': -20, 'lat': -20},
            'zoom': 1},
        title = f"Top 15 remittances inflows to {country}",
        legend_orientation = 'h'
    )
    fig.show()

def outflows_country_2021(country):
    df_country = df[(df.sending_country == country) &
                    (df.mln_remittances > 0.5)].sort_values(
        "mln_remittances", ascending=False
    ).reset_index(drop=True).head(15)
    # df_country.mln_remittances = df_country.mln_remittances.apply(lambda x: np.log(x))

    fig = go.Figure()
    for i in tqdm(range(len(df_country))):
        lons, lats = shortest_path([df_country['lon_start'][i], df_country['lat_start'][i]],
                                   [df_country['lon_end'][i], df_country['lat_end'][i]],
                                   dir=1, n=100)
        color = mpl.colors.rgb2hex(cmap(df_country["mln_remittances"][i] / df_country["mln_remittances"].max()))
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines+markers',
                line=dict(width=2,
                          color=color),
                marker=dict(showscale=True,
                            size=0,
                            color=color,
                            colorscale="spectral",  # [[0, mpl.colors.rgb2hex(cmap(0))],
                            # [1, mpl.colors.rgb2hex(cmap(1))]],
                            cmin=df_country["mln_remittances"].min(),
                            cmax=df_country["mln_remittances"].max()),
                name=df_country["receiving_country"][i],
                hovertext=f"mln USD: {round(df_country['mln_remittances'][i], 2)}"

            )
        )
    fig.update_layout(
        margin={'l': 100, 't': 40, 'b': 100, 'r': 100},
        mapbox={
            'center': {'lon': 10, 'lat': 10},
            'style': "open-street-map",
            'center': {'lon': -20, 'lat': -20},
            'zoom': 1},
        title=f"Top 15 remittances outflows from {country}",
        legend_orientation='h'
    )
    fig.show()

def plot_country(country):
    outflows_country_2021(country)
    inflows_country_2021(country)

plot_country("Austria")
plot_country("Mexico")
plot_country("Germany")


########
# Sankey charts
########

def inflows_outflows_country_sankey(country, show = False):

    try:
        os.mkdir(os.getcwd() + f"\\plots\\country_flows\\{country}")
        out = os.getcwd() + f"\\plots\\country_flows\\{country}\\"
    except:
        out = os.getcwd() + f"\\plots\\country_flows\\{country}\\"

    df.sort_values("mln_remittances", ascending = False, inplace = True)

    #country as origin
    df_or = df[df.sending_country == country].copy()
    labels = df_or.receiving_country.to_list()
    labels.append(country)
    source = [labels.index(country)] * len(df_or)
    target = [x for x in range(len(df_or))]
    value = df_or["mln_remittances"].to_list()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            # color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text=f"Outflow of remittances from {country} in 2021 (mln USD)", font_size=10)
    fig.write_html(out + f"out_remittances_2021.html")
    if show:
        fig.show()

    # country as destination
    df_dest = df[df.receiving_country == country].copy()
    labels = df_dest.sending_country.to_list()
    labels.append(country)
    target = [labels.index(country)] * len(df_dest)
    source = [x for x in range(len(df_dest))]
    value = df_dest["mln_remittances"].to_list()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            # color="red"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text=f"Inflow of remittances in {country} in 2021 (mln USD)", font_size=10)
    fig.write_html(out + f"inflow_remittances_2021.html")
    if show:
        fig.show()

for country in tqdm(df.sending_country.unique(), total = len(df.sending_country.unique()), position=0, leave=True, colour='green', ncols = 80):
    inflows_outflows_country_sankey(country)

#########################
# networks stuff
#######################

#### instantiate the network
df = df[df.mln_remittances > 100]
G = nx.from_pandas_edgelist(df,
                            source = "sending_country",
                            target = "receiving_country",
                            edge_attr = "mln_remittances",
                            create_using=nx.MultiDiGraph) # important to maintain direction of the link

seed = 123 # Seed random number generators for reproducibility
pos = nx.spring_layout(G)

node_sizes = [5 * len(df[df.receiving_country == x]) for x in list(G.nodes)]
M = G.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
cmap = plt.get_cmap('Spectral')

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo", alpha = 0.5)
labels = nx.draw_networkx_labels(G, pos, font_size=7)
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrows = False,
    # arrowstyle="->",
    # arrowsize=6,
    edge_color=edge_colors,
    edge_cmap=cmap,
    width=1
)
# set alpha value for each edge
# for i in range(M):
#     edges[i].set_alpha(edge_alphas[i])

# pc = mpl.collections.PatchCollection(edges, cmap=cmap)
# pc.set_array(edge_colors)

ax = plt.gca()
ax.set_axis_off()
# plt.colorbar(pc, ax=ax)
plt.title("Network of remittances 2021")
plt.show(block = True)

