import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.io as pio
pio.renderers.default = "browser"
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel("C:\\Data\\remittances\\bilateral_remittance_matrix_2021.xlsx",
                   skiprows = 1, nrows = 215)
df = pd.melt(df, id_vars=['sending_country'], value_vars=df.columns.tolist()[1:])
df.rename(columns = {"variable" : "receiving_country", "value" : "mln_remittances"}, inplace = True)

world = gpd.read_file("C:\\Data\\geo\\admin_0\\ne_110m_admin_0_countries.shp")[["ADMIN", "geometry"]]
world.geometry = world.geometry.representative_point()
world['lon'] = world.geometry.x
world['lat'] = world.geometry.y

df = df.merge(world[["ADMIN", "lat", "lon"]], left_on="sending_country", right_on="ADMIN", how="inner")
df.rename(columns = {"lat" : "lat_start", "lon" : "lon_start"}, inplace = True)
df.drop(columns = "ADMIN", inplace = True)
df = df.merge(world[["ADMIN", "lat", "lon"]], left_on="receiving_country", right_on="ADMIN", how="inner")
df.rename(columns = {"lat" : "lat_end", "lon" : "lon_end"}, inplace = True)
df.drop(columns = "ADMIN", inplace = True)

df.fillna(0, inplace = True)

cmap = plt.get_cmap('Reds')

def inflows_country_2021(country):

    df_country = df[(df.receiving_country == country) &
                    (df.mln_remittances > 0.5)].sort_values(
        "mln_remittances", ascending=False
    ).reset_index(drop = True)

    fig = go.Figure()
    for i in tqdm(range(len(df_country))):
        fig.add_trace(
            go.Scattermapbox(
                lon=[df_country['lon_start'][i], df_country['lon_end'][i]],
                lat=[df_country['lat_start'][i], df_country['lat_end'][i]],
                mode='markers+lines',
                # marker=dict(size = 10, colorscale = "inferno",
                #             color = df_country["mln_remittances"][i] / df_country["mln_remittances"].max()),
                line=dict(width=2,
                            color=mpl.colors.rgb2hex(cmap(df_country["mln_remittances"][i] / df_country["mln_remittances"].max()))),
                name = df_country["sending_country"][i],
                hovertext= f"mln USD: {round(df_country['mln_remittances'][i], 2)}"
            )
        )
    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'center': {'lon': 10, 'lat': 10},
            'style': "open-street-map",
            'center': {'lon': -20, 'lat': -20},
            'zoom': 1},
        title = f"Remittances inflows to {country}")
    fig.show()

inflows_country_2021("Italy")

def outflows_country_2021(country):

    df_country = df[(df.sending_country == country) &
                    (df.mln_remittances > 0.5)].sort_values(
        "mln_remittances", ascending=False
    ).reset_index(drop = True)

    fig = go.Figure()
    for i in tqdm(range(len(df_country))):
        fig.add_trace(
            go.Scattermapbox(
                lon=[df_country['lon_start'][i], df_country['lon_end'][i]],
                lat=[df_country['lat_start'][i], df_country['lat_end'][i]],
                mode='markers+lines',
                # marker=dict(size = 10, colorscale = "inferno",
                #             color = df_country["mln_remittances"][i] / df_country["mln_remittances"].max()),
                line=dict(width=2,
                            color=mpl.colors.rgb2hex(cmap(df_country["mln_remittances"][i] / df_country["mln_remittances"].max()))),
                name = df_country["receiving_country"][i],
                hovertext= f"mln USD: {round(df_country['mln_remittances'][i], 2)}"
            )
        )
    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'center': {'lon': 10, 'lat': 10},
            'style': "open-street-map",
            'center': {'lon': -20, 'lat': -20},
            'zoom': 1},
        title = f"Remittances outflows from {country}")
    fig.show()

outflows_country_2021("Italy")

#### instantiate the network
df = df[df.mln_remittances > 0.5]
G = nx.from_pandas_edgelist(df,
                            source = "sending_country",
                            target = "receiving_country",
                            edge_attr = "mln_remittances",
                            create_using=nx.MultiDiGraph) # important to maintain direction of the link


seed = 123 # Seed random number generators for reproducibility
pos = nx.circular_layout(G)

node_sizes = [100 * df[df.receiving_country == x].mln_remittances.sum() / df.mln_remittances.sum() for x in list(G.nodes)]
M = G.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
cmap = plt.get_cmap('Reds')

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrowstyle="->",
    arrowsize=1,
    edge_color=edge_colors,
    edge_cmap=cmap,
    width=1,
)
# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

pc = mpl.collections.PatchCollection(edges, cmap=cmap)
pc.set_array(edge_colors)

ax = plt.gca()
ax.set_axis_off()
plt.colorbar(pc, ax=ax)
plt.show(block = True)

##find pagerank of the network
pr = nx.pagerank(G, weight="mln_remittances")

df_pr = pd.DataFrame.from_dict({"country" : [x for x in pr.keys()], "pagerank" : [pr[x] for x in pr.keys()]})
df_tot_send = df[["sending_country", "mln_remittances"]].groupby("sending_country").sum().reset_index().rename(columns = {"sending_country" : "country"})
df_pr = df_pr.merge(df_tot_send, on = "country").rename(columns = {"mln_remittances" : "sent_remittances"})
df_tot_rec = df[["receiving_country", "mln_remittances"]].groupby("receiving_country").sum().reset_index().rename(columns = {"receiving_country" : "country"})
df_pr = df_pr.merge(df_tot_rec, on = "country").rename(columns = {"mln_remittances" : "received_remittances"})
# df_pr.received_remittances = 200 * df_pr.received_remittances / df_pr.received_remittances.sum()

##import migration stock data

df_stock = pd.read_excel("C://Data//general//migration_stock_abs.xls")
df_stock.rename(columns = {"Country Name" : "country"}, inplace = True)
df_pr = df_pr.merge(df_stock[["country", "2015"]], on="country", how="left")
df_pr.rename(columns = {"2015" : "migrant_pop"}, inplace = True)

# regression
x = df_pr['migrant_pop'].to_numpy().reshape(-1, 1) /1_000_000
y = df_pr['sent_remittances'].to_numpy() / 1000
reg = LinearRegression().fit(x, y)
df_pr['reg'] = reg.predict(x)

##plot
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_pr.migrant_pop / 1_000_000,
                         y = df_pr.sent_remittances / 1000,
                         mode="markers",
                         marker = dict(size = 10),
                         customdata= df_pr["country"],
                         hovertemplate=
                         "<b>%{customdata}</b><br>" +
                         "Migrant population hosted: %{x:.2f}mln people<br>" +
                         "Remittances sent: %{y:.2f}bn$,<br>" +
                         # "Life Expectancy: %{y:.0f}<br>"
                         "<extra></extra>",
                         showlegend = False
                         )
              )
fig.add_trace(go.Scatter(x = df_pr.migrant_pop / 1_000_000,
              y = df_pr["reg"],
              mode = "lines",
              line = dict(color = "red"),
              name = "fitted line"
                         ))
fig.update_layout(title = "Migrant population hosted vs. remittances sent")
fig.update_xaxes(title="Total migrant population hosted (mln people, 2015)")
fig.update_yaxes(title="Sent remittances (bn USD, 2021)")
fig.write_html("plots//migrant_pop_v_remittances_sent_WITH_USA.html")
fig.show()


df_no_usa = df_pr[df_pr.country != "United States of America"]
# regression
x = df_no_usa['migrant_pop'].to_numpy().reshape(-1, 1) /1_000_000
y = df_no_usa['sent_remittances'].to_numpy() / 1000
reg = LinearRegression().fit(x, y)
df_no_usa['reg'] = reg.predict(x)

fig = go.Figure()
fig.add_trace(go.Scatter(y = df_no_usa.sent_remittances / 1000,
                         x = df_no_usa.migrant_pop / 1_000_000,
                         mode="markers",
                         marker = dict(size = 10),
                         customdata= df_no_usa["country"],
                         hovertemplate=
                         "<b>%{customdata}</b><br>" +
                         "Migrant population hosted: %{x:.2f}mln people<br>" +
                         "Remittances sent: %{y:.2f}bn$,<br>" +
                         # "Life Expectancy: %{y:.0f}<br>"
                         "<extra></extra>",
                         showlegend = False
                         )
              )
fig.add_trace(go.Scatter(x = df_no_usa.migrant_pop / 1_000_000,
              y = df_no_usa["reg"],
              mode = "lines",
              line = dict(color = "red"),
              name = "fitted line"
                         ))
fig.update_layout(title = "Migrant population hosted vs. remittances sent (excluding USA)")
fig.update_xaxes(title="Total migrant population hosted (mln people, 2015)")
fig.update_yaxes(title="Sent remittances (bn USD, 2021)")
fig.write_html("plots//migrant_pop_v_remittances_sent_NO_USA.html")
fig.show()