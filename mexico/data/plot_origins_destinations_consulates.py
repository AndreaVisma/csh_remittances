import pandas as pd
import os
from tqdm import tqdm
import time
import geopandas
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from utils import *
import itertools
from itertools import permutations

mapbox_access_token = open("c:\\data\\geo\\mapbox_token.txt").read()

df = pd.read_csv(os.getcwd() + "\\mexico\\data\\consulates_network_with_geo.csv")

### aggregate by state of origin, destination, year
df_agg = df[['Número de Matrículas','registration_consulate', 'year',
       'consul_lat', 'consul_lon', 'state', 'state_lat', 'state_lon']].groupby(
    ['year', 'state', 'registration_consulate']
).agg({
    'Número de Matrículas' : 'sum',
    'consul_lat' : 'first', 'consul_lon' : 'first',
    'state_lat' : 'first', 'state_lon' : 'first'
}).reset_index()

### aggregate by state of origin, year
df_agg_state = df[['Número de Matrículas', 'year', 'state']].groupby(
    ['year', 'state']
).agg({
    'Número de Matrículas' : 'sum'
}).reset_index()

gdf = geopandas.read_file("c:\\data\\geo\\world_admin2\\World_Administrative_Divisions.shp")
gdf = gdf[(gdf.COUNTRY == "Mexico") & (gdf.LAND_RANK == 5)][['NAME', 'geometry']]
gdf.sort_values('NAME', inplace = True)
gdf.rename(columns = {'NAME' : 'state'}, inplace = True)

miss = ['Coahuila de Zaragoza','Veracruz de Ignacio de la Llave', 'Michoacán de Ocampo','México']
fix = ['Coahuila', 'Veracruz', 'Michoacán', 'Estado de México']
dict_names = dict(zip(miss, fix))
gdf.loc[gdf.state.isin(miss), 'state'] = (
    gdf.loc[gdf.state.isin(miss), 'state'].map(dict_names))

df_agg_state = df_agg_state.merge(gdf[['state', 'geometry']], left_on='state',
              right_on = 'state', how = 'left')
df_agg_state = geopandas.GeoDataFrame(df_agg_state, geometry='geometry')

###########
# plots
##########
cmap = plt.get_cmap('YlGn')

def mapbox_emigration_states_per_year(year, show = False):

    outfolder = os.getcwd() + "\\mexico\\data\\plots\\mapbox_per_year\\"
    df_year = df_agg_state[df_agg_state.year == year].copy()
    fig = go.Figure()
    fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df_year['geometry'].to_json()),
                                            locations=df_year.index,
                                            z=df_year["Número de Matrículas"],
                                            text=df_year['state'],
                                            hovertemplate=
                                            '<br>State: %{text}' +
                                            '<br>Registered migrants from<br>the state: %{z:,.0f}',
                                            colorscale="speed", marker_opacity=0.7,
                                            colorbar_title="Number of registered<br>migrants form the state"))
    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=23.5,
                lon=-99
            ),
            pitch=0,
            zoom=4.6
        )
    )
    fig.update_layout(title=f'Registered migrants to the US per Mexican state, {year}')
    fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
    fig.write_html(outfolder + f"\\mapbox_migrants_{year}.html")
    if show:
        fig.show()

for year in tqdm(df_agg_state.year.unique()):
    mapbox_emigration_states_per_year(year)

def network_per_year_state_granular(year, mexican_state):

    df_year = df[(df.year == year) &
                 (df["Estado de Origen"] == mexican_state)].copy().reset_index(drop = True)

    fig = go.Figure()
    for i in tqdm(range(len(df_year))):
        lons, lats = shortest_path([df_year['mun_lon'][i], df_year['mun_lat'][i]],
                                   [df_year['consul_lon'][i], df_year['consul_lat'][i]],
                                   dir = 1, n=50)
        color = mpl.colors.rgb2hex(cmap(df_year['Número de Matrículas'][i] / df_year['Número de Matrículas'].max()))
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines+markers',
                line=dict(width=20 * df_year["Número de Matrículas"][i] / df_year["Número de Matrículas"].max(),
                          color=color),
                marker = dict(showscale=True,
                              size = 0,
                              color = color,
                              colorscale= "YlGn",
                              cmin=df_year["Número de Matrículas"].min(),
                              cmax=df_year["Número de Matrículas"].max()),
                hovertext= f"Numero matriculas: {df_year['Número de Matrículas'][i]}" +
                f"<br>Origin: {df_year['mun_name'][i]}" +
                f"<br>Destination consulate: {df_year['registration_consulate'][i]}",
                showlegend=False
            )
        )
    fig.update_layout(
        title = f"Mexican migrants from {mexican_state} in {year}",
        hovermode='closest',
        mapbox=dict(
            style = "open-street-map",
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=32,
                lon=-96
            ),
            pitch=0,
            zoom=3
        )
    )
    fig.update_traces(opacity=.5)
    fig.show()

def network_per_year_whole_states_all(year, mexican_state, show = False):

    outfolder = os.getcwd() + "\\mexico\\data\\plots\\"
    try:
        os.mkdir(outfolder + f"{mexican_state}")
    except:
        time.sleep(0.0000001)
    outfolder = outfolder + f"{mexican_state}\\"

    df_agg_state = df_agg[(df_agg.year == year) &
                 (df_agg["state"] == mexican_state)].copy().reset_index(drop=True)

    fig = go.Figure()
    for i in tqdm(range(len(df_agg_state))):
        lons, lats = shortest_path([df_agg_state['state_lon'][i], df_agg_state['state_lat'][i]],
                                   [df_agg_state['consul_lon'][i], df_agg_state['consul_lat'][i]],
                                   dir=1, n=50)
        color = mpl.colors.rgb2hex(cmap(df_agg_state['Número de Matrículas'][i] / df_agg_state['Número de Matrículas'].max()))
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines+markers',
                line=dict(width=2, #0 * df_agg_state["Número de Matrículas"][i] / df_agg_state["Número de Matrículas"].max(),
                          color=color),
                marker=dict(showscale=True,
                            size=0,
                            color=color,
                            colorscale="YlGn",
                            cmin=df_agg_state["Número de Matrículas"].min(),
                            cmax=df_agg_state["Número de Matrículas"].max(),
                            colorbar=dict(
                                title='Registered matriculas')
                            ),
                hovertext=f"Numero matriculas: {df_agg_state['Número de Matrículas'][i]}" +
                          f"<br>Origin: {df_agg_state['state'][i]}" +
                          f"<br>Destination consulate: {df_agg_state['registration_consulate'][i]}",
                showlegend=False
            )
        )
    fig.update_layout(
        title=f"Mexican migrants from {mexican_state} in {year}",
        hovermode='closest',
        mapbox=dict(
            style="open-street-map",
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=32,
                lon=-96
            ),
            pitch=0,
            zoom=3
        )
    )
    fig.update_traces(opacity=.7)
    fig.write_html(outfolder + f"{mexican_state}_flows_{year}.html")
    fig.show() if show else time.sleep(0.00000001)


network_per_year_whole_states_all(2022, 'Campeche', show=True)

for year in tqdm(df_agg.year.unique()):
    for state in tqdm(df_agg.state.unique()):
        network_per_year_whole_states_all(year, state, show=False)

def sankey_year(year, show = False):
    # country as origin
    outfolder = os.getcwd() + "\\mexico\\data\\plots\\"

    df_or = df_agg[(df_agg.year == year)].copy().reset_index(drop=True)

    labels = df_or.state.unique().tolist() + df_or.registration_consulate.unique().tolist()

    source = [labels.index(row['state']) for ind,row in df_or.iterrows()]
    target = [labels.index(row['registration_consulate']) for ind,row in df_or.iterrows()]
    value = df_or['Número de Matrículas'].to_list()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text=f"Distribution of Mexican-born migrants<br>"
                                 f"registered at a consulate in {year}", font_size=10)
    fig.write_html(outfolder + f"sankey_migrants_{year}.html")
    if show:
        fig.show()

def sankey_states_comparison(year, states, show = False):
    # country as origin
    outfolder = os.getcwd() + "\\mexico\\data\\plots\\comparison\\"

    df_or = (df_agg[(df_agg.year == year) & (df_agg.state.isin(states))].
             copy().reset_index(drop=True).
             sort_values("Número de Matrículas", ascending = False))

    labels = (df_or.state.unique().tolist() +
              df_or.loc[~df_or.duplicated('registration_consulate'), "registration_consulate"].tolist())

    source = [labels.index(row['state']) for ind,row in df_or.iterrows()]
    target = [labels.index(row['registration_consulate']) for ind,row in df_or.iterrows()]
    value = df_or['Número de Matrículas'].to_list()

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

    fig.update_layout(title_text=f"Distribution of Mexican-born migrants<br>"
                                 f"registered at a consulate in {year}<br>"
                      f"and coming from {','.join(states)}", font_size=10)
    fig.write_html(outfolder + f"sankey_{year}_comparison_{', '.join(states)}.html")
    if show:
        fig.show()

sankey_states_comparison(2022, ['Veracruz', 'Jalisco'],True)
sankey_states_comparison(2018, ['Veracruz', 'Jalisco'],True)

sankey_states_comparison(2022, ['Veracruz', 'Jalisco', 'Nuevo León'],True)

def pie_from_state_year(year, state, show = False):
    outfolder = os.getcwd() + "\\mexico\\data\\plots\\"
    try:
        os.mkdir(outfolder + f"{state}")
    except:
        time.sleep(0.0000001)
    outfolder = outfolder + f"{state}\\"

    df_or = (df_agg[(df_agg.year == year) & (df_agg.state == state)].
             copy().reset_index(drop=True).
             sort_values("Número de Matrículas", ascending = False))

    fig = px.pie(df_or, values="Número de Matrículas",
                 names='registration_consulate',
                 title=f'Population registred in each consulate<br>coming from {state}',
                 color_discrete_sequence=px.colors.sequential.speed)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.write_html(outfolder + f"piechart_{year}.html")
    if show:
        fig.show()

for state in df.state.unique():
    pie_from_state_year(2022, state, show = False)

