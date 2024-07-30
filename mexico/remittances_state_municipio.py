"""
Script: mexico_remittances_explore.py
Author: Andrea Vismara
Date: 29/07/2024
Description: Explores the data for the remittances inflow in mexico, at state and municipio level
"""

##imports
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

out_folder = "c:\\git-projects\\csh_remittances\\mexico\\plots\\summary_remittances\\"

##load the data in
df = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_state_municipio.xlsx",
                   skiprows=9)
df = df.iloc[8:,:]

df = pd.melt(df, id_vars="Título", value_vars=df.columns.tolist()[1:], value_name="mln_USD_remesas")
df.rename(columns = {"Título": "Three months period starting"}, inplace = True)
df['entity_type'] = 'state'
df.loc[df.variable.str.contains('Municipio'), "entity_type"] = 'municipio'
df['state'] = None
df.loc[df.entity_type == 'state', "state"] = df.loc[df.entity_type == 'state'].variable.apply(lambda x: x.split(', ')[1])
df['state'].ffill(inplace=True)
df['municipio'] = None
df.loc[df.entity_type == 'municipio', "municipio"] = df.loc[df.entity_type == 'municipio'].variable.apply(lambda x: x.split(', ')[3])

df.drop(columns = 'variable', inplace = True)

df.to_excel("c:\\data\\remittances\\mexico\\remittances_municipio_long.xlsx", index = False)

##plot remittances received by state
gdf = geopandas.read_file("c:\\data\\geo\\world_admin2\\World_Administrative_Divisions.shp")
gdf = gdf[(gdf.COUNTRY == "Mexico") & (gdf.LAND_RANK == 5)][['NAME', 'geometry']]
gdf.sort_values('NAME', inplace = True)
gdf.rename(columns = {'NAME' : 'state'}, inplace = True)

miss = ['Coahuila de Zaragoza','Veracruz de Ignacio de la Llave', 'Michoacán de Ocampo','México']
fix = ['Coahuila', 'Veracruz', 'Michoacán', 'Estado de México']
dict_names = dict(zip(miss, fix))
gdf.loc[gdf.state.isin(miss), 'state'] = (
    gdf.loc[gdf.state.isin(miss), 'state'].map(dict_names))

df_state = df[df.entity_type == 'state'].merge(gdf, on='state')
df_state = geopandas.GeoDataFrame(df_state, geometry = 'geometry')


fig = px.line(df_state, x="Three months period starting", y = 'mln_USD_remesas', color = 'state')
fig.update_layout(title = "Remittances received by Mexican state, mln USD")
fig.write_html(os.getcwd() + "\\mexico\\plots\\remittances_per_state_overtime.html")
fig.show()
###########
# plots
##########
cmap = plt.get_cmap('YlGn')

def mapbox_remittances_states_per_year(date_, show = False):

    outfolder = os.getcwd() + "\\mexico\\plots\\geo_breakdown\\"
    year = date_.year
    quarter = 1 + (date_.month-1)//3
    df_year = df_state[df_state["Three months period starting"] == date_].copy()
    fig = go.Figure()
    fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df_year['geometry'].to_json()),
                                            locations=df_year.index,
                                            z=df_year["mln_USD_remesas"],
                                            text=df_year['state'],
                                            hovertemplate=
                                            '<br>State: %{text}' +
                                            '<br>Total remittances, from<br>the state: %{z:,.0f} mln USD',
                                            colorscale="speed", marker_opacity=0.7,
                                            colorbar_title="Mln USD<br>in remittances"))
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
    fig.update_layout(title=f'Remittances flow to Mexico during Q{quarter} {year}')
    fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
    fig.write_html(outfolder + f"\\state_remittances_{year}Q{quarter}.html")
    if show:
        fig.show()

for date_ in tqdm(df_state["Three months period starting"].unique()):
    mapbox_remittances_states_per_year(date_)