import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import os
import matplotlib.ticker as mtick
import geopandas
from utils import *
import plotly.io as pio
pio.renderers.default = 'browser'

out_folder = "c:\\git-projects\\csh_remittances\\mexico\\models\\plots\\"

mapbox_access_token = open("c:\\data\\geo\\mapbox_token.txt").read()
##load geo coordinates
gdf = geopandas.read_file("c:\\data\\geo\\world_admin2\\World_Administrative_Divisions.shp")
gdf = gdf[(gdf.COUNTRY == "Mexico") & (gdf.LAND_RANK == 5)][['NAME', 'geometry']]
gdf.sort_values('NAME', inplace = True)
gdf.rename(columns = {'NAME' : 'state'}, inplace = True)

miss = ['Coahuila de Zaragoza','Veracruz de Ignacio de la Llave', 'Michoacán de Ocampo','México']
fix = ['Coahuila', 'Veracruz', 'Michoacán', 'Estado de México']
dict_names = dict(zip(miss, fix))
gdf.loc[gdf.state.isin(miss), 'state'] = (
    gdf.loc[gdf.state.isin(miss), 'state'].map(dict_names))
###

df_ime = pd.read_excel("c:\\data\\migration\\mexico\\migrants_mex_state_aggregate.xlsx")
df_ime = df_ime[['nr_registered', 'mex_state', 'year']].groupby(['mex_state', 'year'], as_index = False).sum()

df_conapo = pd.read_csv("c:\\data\\migration\\mexico\\conapo\\iim_base2020e.csv")
df_conapo.rename(columns=dict(zip(df_conapo.columns, ['year', 'mex_state', 'tot_hh', 'hh_reciben_remesas',
                                                      'hh_with_resident_migrants_us', 'hh_with_circular_migrants_us',
                                                      'hh_with_return_migrants_us', 'hh_with_return_migrants_us',
                                                      'ind_int_mig', 'grade_int_mig', 'rank'])), inplace = True)
df_conapo['year'] = 2020
fix = ['Coahuila', 'Estado de México', 'Michoacán', 'Querétaro', 'Veracruz']
miss = ['Coahuila de Zaragoza', 'México', 'Michoacán de Ocampo', 'Querétaro de Arteaga', 'Veracruz de Ignacio de la Llave']
dict_names = dict(zip(miss, fix))
df_conapo.loc[df_conapo.mex_state.isin(miss), 'mex_state'] = df_conapo.loc[df_conapo.mex_state.isin(miss), 'mex_state'].map(dict_names)

df_rem = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_state_long.xlsx")
df_rem.rename(columns = {df_rem.columns[0]: "quarter"}, inplace = True)
df_rem = df_rem[['quarter', 'mln_USD_remesas', 'state', 'geometry']].groupby([df_rem.quarter.dt.year, 'state']).agg(
    {'mln_USD_remesas' : 'sum',
     'geometry' : 'first'}
).reset_index()
df_rem.rename(columns = {'quarter' : 'year'}, inplace = True)

df = df_ime.merge(df_conapo, on=['mex_state', 'year']).rename(columns = {'mex_state':'state'}).merge(
    df_rem[df_rem.year == 2020], on = ['state', 'year']
).drop(columns = 'geometry')
df = df.merge(gdf[['state', 'geometry']], left_on='state',
              right_on = 'state', how = 'left')
df = geopandas.GeoDataFrame(df, geometry='geometry')

##plots
## total households
outfolder = os.getcwd() + "\\mexico\\plots\\"
fig = go.Figure()
fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df['geometry'].to_json()),
                                        locations=df.index,
                                        z=df["tot_hh"],
                                        text=df['state'],
                                        hovertemplate=
                                        '<br>State: %{text}' +
                                        '<br>Total nr households: %{z:,.0f}',
                                        colorscale="speed", marker_opacity=0.7,
                                        colorbar_title="Total number<br>households (2020)"))
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
fig.update_layout(title=f'Total number of households, CONAPO, by Mexican state')
fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
fig.write_html(outfolder + f"\\tot_households_per_state_2020.html")
fig.show()

##remittances
df['hh_reciben_remesas_tot'] = 0.01 * df['hh_reciben_remesas'] * df.tot_hh

outfolder = os.getcwd() + "\\mexico\\plots\\"
fig = go.Figure()
fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df['geometry'].to_json()),
                                        locations=df.index,
                                        z=df["hh_reciben_remesas_tot"],
                                        text=df['state'],
                                        hovertemplate=
                                        '<br>State: %{text}' +
                                        '<br>Nr households receiving remittances: %{z:,.0f}',
                                        colorscale="speed", marker_opacity=0.7,
                                        colorbar_title="Total number<br>households (2020)"))
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
fig.update_layout(title=f'Number of households receiving remittances, CONAPO, by Mexican state')
fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
fig.write_html(outfolder + f"\\households_receiving_remittances_per_state_2020.html")
fig.show()

##registered migrants v survey
df['mig_hh'] = (0.01 * df['hh_with_resident_migrants_us'] + 0.01 * df['hh_with_circular_migrants_us']) * df.tot_hh

outfolder = os.getcwd() + "\\mexico\\plots\\"
fig = go.Figure()
fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df['geometry'].to_json()),
                                        locations=df.index,
                                        z=df["mig_hh"],
                                        text=df['state'],
                                        hovertemplate=
                                        '<br>State: %{text}' +
                                        '<br>Nr households with US migrants: %{z:,.0f}',
                                        colorscale="speed", marker_opacity=0.7,
                                        colorbar_title="Total number<br>households (2020)"))
fig.update_layout(

    hovermode='closest',
    mapbox=dict(accesstoken=mapbox_access_token, bearing=0,
        center=go.layout.mapbox.Center(lat=23.5, lon=-99), pitch=0,zoom=4.6))
fig.update_layout(title=f'Number of households with at least 1 member migrated to the US, CONAPO, by Mexican state')
fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
fig.write_html(outfolder + f"\\households_with_US_migrants_per_state_2020.html")
fig.show()

#scatterplot
fig = px.scatter(df, x="mig_hh", y="nr_registered", hover_data=['state'], trendline="ols")
fig.update_yaxes(title = "nr registered IME")
fig.update_xaxes(title = "Households with migrants, CONAPO")
fig.update_layout(title=f'Migration sources by Mexican state, CONAPO v. IME')
fig.write_html(outfolder + f"\\CONAPO_v_IME_migrants_2020_TRENDLINE.html")
fig.show()

fig = px.scatter(df, x="mig_hh", y="nr_registered", hover_data=['state'], size='mln_USD_remesas', color='state')
fig.update_yaxes(title = "nr registered IME")
fig.update_xaxes(title = "Households with migrants, CONAPO")
fig.update_layout(title=f'Migration sources by Mexican state, CONAPO v. IME')
fig.write_html(outfolder + f"\\CONAPO_v_IME_migrants_2020_states.html")
fig.show()

##
df_all_years = df_ime.merge(df_conapo, on=['mex_state']).rename(columns = {'mex_state':'state'}).merge(
    df_rem, on = ['state']
).drop(columns = 'geometry')
df_all_years = df_all_years.merge(gdf[['state', 'geometry']], left_on='state',
              right_on = 'state', how = 'left')
df_all_years = geopandas.GeoDataFrame(df_all_years, geometry='geometry')
df_all_years['mig_hh'] = (0.01 * df_all_years['hh_with_resident_migrants_us'] + 0.01 * df_all_years['hh_with_circular_migrants_us']) * df_all_years.tot_hh

for year in df_all_years.year_x.unique():
    fig = px.scatter(df_all_years[df_all_years.year_x == year], x="mig_hh", y="nr_registered", hover_data=['state'], trendline="ols")
    fig.update_yaxes(title = "nr registered IME")
    fig.update_xaxes(title = "Households with migrants, CONAPO")
    fig.update_layout(title=f'Migration sources by Mexican state, {year}, CONAPO v. IME')
    fig.write_html(outfolder + f"\\IMEvCONAPO\\CONAPO_v_IME_migrants_{year}_states.html")
    # fig.show()

## expected migrants v CONAPO households
df['expected_migrants'] = df['mln_USD_remesas'] * 1_000_000 / 12 / 326 #assuming one sending of remittances a month
fig = px.scatter(df, x="mig_hh", y="expected_migrants", hover_data=['state'],
                 trendline="ols", text="state")
fig.update_xaxes(title="Households with migrants CONAPO")
fig.update_yaxes(title="Expected migrants given remittances")
fig.update_layout(title=f'Expected migrants given remittances v CONAPO migrants data')
fig.write_html(outfolder + f"\\CONAPO_v_expected_migrants_2022_states.html")
fig.show()