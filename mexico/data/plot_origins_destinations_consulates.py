import pandas as pd
import os
import time
from geopy.geocoders import Nominatim
from tqdm import tqdm
import geopandas
import ast
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
from utils import *

mapbox_access_token = open("c:\\data\\geo\\mapbox_token.txt").read()

###
gdf = geopandas.read_file("C:\\data\\geo\\georef-mexico-municipality-millesime@public\\georef-mexico-municipality-millesime.shp")
gdf['mun_name'] = gdf['mun_name'].apply(lambda x: ast.literal_eval(x)[0])
gdf.sort_values('mun_name', inplace = True)
gdf = gdf[~gdf.duplicated(['mun_name'])]

# Load the uploaded Excel file to examine its contents
file_path = os.getcwd() + "\\mexico\\data\\consulates_network_2018_2022.xlsx"
df = pd.read_excel(file_path)
df.rename(columns = {'Municipio de Origen': 'mun_name'}, inplace = True)

miss = ['Batopilas','Heroica Villa Tezoatlán de Segura y Luna, Cuna de la Independencia de Oaxaca',
        'Magdalena Jicotlán', 'MonteMorelos', 'San Juan Mixtepec - Distr. 08 -', 'San Juan Mixtepec - Distr. 26 -',
        'San Pedro Mixtepec - Distr. 22 -','San Pedro Mixtepec - Distr. 26 -', 'San Pedro Totolapa',
        'Silao', 'Temósachi', 'Ticumuy', 'Tixpéhual ', 'Villa de Tututepec de Melchor Ocampo',
        'Zacatepec de Hidalgo', 'Zapotitlán del Río']
fix = ['Batopilas de Manuel Gómez Morín', 'Heroica Villa Tezoatlán de Segura y Luna', 'Santa Magdalena Jicotlán',
       'Montemorelos', 'San Juan Mixtepec','San Juan Mixtepec', 'San Juan Mixtepec', 'San Juan Mixtepec',
       'San Pedro Totolápam', 'Silao de la Victoria', 'Temósachic', 'Timucuy',
       'Tixpéhual', 'Melchor Ocampo', 'Zacatepec', 'San Antonio Huitepec']
dict_names = dict(zip(miss, fix))

df.loc[df.mun_name.isin(miss), 'mun_name'] = df.loc[df.mun_name.isin(miss), 'mun_name'].map(dict_names)

#merge
df = df.merge(gdf[['mun_name', 'geometry']], on='mun_name', how = 'left')
df = geopandas.GeoDataFrame(df, geometry="geometry")
df['geometry'] = df['geometry'].centroid
df.rename(columns = {'geometry' : 'mun_geometry'}, inplace = True)
df['mun_lat'] = df.mun_geometry.y
df['mun_lon'] = df.mun_geometry.x

##now coordinates for US states
gdf = geopandas.read_file("c:\\data\\geo\\us-major-cities\\USA_Major_Cities.shp")
gdf = gdf[['NAME', 'geometry']].copy()
gdf.sort_values('NAME', inplace = True)
gdf.rename(columns = {'NAME' : 'registration_consulate'}, inplace = True)

miss = list(set(df.registration_consulate) - set(gdf.registration_consulate))
fix = ['San Jose', 'Del Rio', 'Indianapolis', 'Calexico',
       'Philadelphia', 'Boise City', 'Minneapolis',
       'New Orleans', 'New York', 'Los Angeles', 'McAllen',
       'Presidio']
dict_names = dict(zip(miss, fix))

df.loc[df.registration_consulate.isin(miss), 'registration_consulate'] = (
    df.loc[df.registration_consulate.isin(miss), 'registration_consulate'].map(dict_names))

df = df.merge(gdf[['registration_consulate', 'geometry']], on='registration_consulate', how = 'left')
df = geopandas.GeoDataFrame(df, geometry="geometry")
df['geometry'] = df['geometry'].centroid
df.rename(columns = {'geometry' : 'consul_geometry'}, inplace = True)

#fill in by hand the coordinates for Presidio
df.loc[df.registration_consulate == 'Presidio', 'consul_geometry'] = [geopandas.points_from_xy(x=[-104.368], y=[29.563])]

df['consul_lat'] = df.consul_geometry.y
df['consul_lon'] = df.consul_geometry.x

df.isna().sum()

###########
# plots
##########
cmap = plt.get_cmap('YlGn')

def network_per_year_state(year, mexican_state):

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

network_per_year_state(2022, 'Campeche')
network_per_year_state(2022, 'Baja California')
network_per_year_state(2022, 'Ciudad de México')
