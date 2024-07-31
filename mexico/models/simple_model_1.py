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
##

df = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_seasonally_adjusted.xlsx")

ex_rate = pd.read_excel("c:\\data\\economic\\mexico\\peso_usd_exrate.xls",
                        skiprows = 10)
ex_rate = ex_rate.groupby([ex_rate.observation_date.dt.month, ex_rate.observation_date.dt.year]).head(1).reset_index(drop = True)
ex_rate["observation_date"] = ex_rate["observation_date"].apply(lambda x: datetime.datetime(x.year, x.month, 1, 0, 0))
ex_rate.rename(columns={"observation_date":"date", "DEXMXUS" : "pesos_for_dollar"}, inplace = True)
ex_rate.loc[ex_rate.pesos_for_dollar == 0, "pesos_for_dollar"] = np.nan
ex_rate["pesos_for_dollar"] = ex_rate["pesos_for_dollar"].interpolate()
df = df.merge(ex_rate, on = "date", how= "left")
df["total_remittances_pesos"] = df.total_mln_seas * df.pesos_for_dollar
df["average_remittance_pesos"] = df.total_mean_op_seas * df.pesos_for_dollar

###
promedio_peso = df.average_remittance_pesos.mean()
promedio_usd = df.total_mean_op_seas.mean()

df["N_est"] = 1_000 * df["total_mln_seas"] / promedio_usd
f = df["N_est"].plot(label = "N_est")
df["total_operations_seas"].plot(ax = f, label = "N_report")
plt.legend()
plt.show(block = True)

####
data_folder = os.getcwd() + "\\data_downloads\\data\\"

#diaspora data
df_all = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
                       skiprows=10, usecols="B,F,H:N", sheet_name="Table 1")
df_all.rename(columns={'Region, development group, country or area of destination': 'destination',
                   'Region, development group, country or area of origin': 'origin'}, inplace=True)
df_all['origin'] = df_all['origin'].apply(lambda x: remove_asterisc(x))
df_all['origin'] = clean_country_series(df_all['origin'])
df_all['destination'] = df_all['destination'].apply(lambda x: remove_asterisc(x))
df_all['destination'] = clean_country_series(df_all['destination'])
df_all.dropna(inplace = True)

df_mex = df_all[(df_all.origin == "Mexico") & (df_all.destination == "USA")]
df_mex = pd.melt(df_mex, id_vars=['origin', 'destination'], value_vars=[x for x in df_mex.columns[2:]],
                 var_name='year', value_name='migrants_stock')
df_mex = df_mex[df_mex.year > 1990]
df_mex.loc[len(df_mex.index) + 1] = ['Mexico', 'USA', 2022, 10_820_514]

df_usa = df_all[(df_all.origin == "USA") & (df_all.destination == "Mexico")]
df_usa = pd.melt(df_usa, id_vars=['origin', 'destination'], value_vars=[x for x in df_usa.columns[2:]],
                 var_name='year', value_name='migrants_stock')

### agggregate remittances by year
df_agg = df.groupby(df.date.dt.year, as_index = False).mean()
df_agg['year'] = df_agg.date.dt.year

df_mex = df_mex.merge(df_agg[["year", "N_est", "total_operations_seas"]], on = "year", how = "left")
df_mex = df_mex[['year', 'migrants_stock', 'N_est', 'total_operations_seas']]
df_mex.rename(columns = {'migrants_stock' : 'mexican_migrants'}, inplace = True)

df_mex = df_mex.merge(df_usa[['year', 'migrants_stock']], on = "year", how = "left")
df_mex.rename(columns = {'migrants_stock' : 'us_migrants'}, inplace = True)
df_mex = df_mex.interpolate()
df_mex["N_est"] = df_mex["N_est"]
df_mex["total_operations_seas"] = df_mex["total_operations_seas"]

f = df_mex["N_est"].plot(label = "N_est")
df_mex["us_migrants"].plot(ax = f, label = "us_migrants")
df_mex["mexican_migrants"].plot(ax = f, label = "mexican_migrants")
plt.legend()
plt.show(block = True)

##if only mexicans sent money
df_mex["alpha"] = 100 * df_mex["total_operations_seas"] * 1000 / df_mex["mexican_migrants"]
ax = df_mex.plot("year", "alpha")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title("Percentage of mexican migrant population \n sending remittances each month")
plt.grid(True)
plt.show(block = True)

## if only americans sent money
df_mex["beta"] = 100 * df_mex["total_operations_seas"] * 1000 / df_mex["us_migrants"]
ax = df_mex.plot("year", "beta")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title("Percentage of american migrant population \n sending remittances to MEX each month")
plt.grid(True)
plt.show(block = True)

##################################
## test the change of the percentage by state
##################################
df_mig = pd.read_excel("c:\\data\\migration\\mexico\\migrants_mex_state_aggregate.xlsx")
df_mig = df_mig[['nr_registered', 'mex_state', 'year']].groupby(['mex_state', 'year'], as_index = False).sum()

df_rem = pd.read_excel("c:\\data\\remittances\\mexico\\remittances_state_long.xlsx")
df_rem.rename(columns = {df_rem.columns[0]: "quarter"}, inplace = True)
df_rem = df_rem[['quarter', 'mln_USD_remesas', 'state', 'geometry']].groupby([df_rem.quarter.dt.year, 'state']).agg(
    {'mln_USD_remesas' : 'sum',
     'geometry' : 'first'}
).reset_index()

###check for 2022
tot_mig_2022 = df_mex[df_mex.year == 2022].mexican_migrants.item()
df_mig_2022 = df_mig[df_mig.year == 2022].copy()
df_mig_2022['share'] = df_mig_2022['nr_registered'] / df_mig_2022['nr_registered'].sum()
df_mig_2022['est_migrants'] = df_mig_2022['share'] * tot_mig_2022

df_22 = df_mig_2022.merge(df_rem[df_rem.quarter == 2022], right_on = 'state', left_on = 'mex_state')
df_22.drop(columns = ['mex_state', 'quarter', 'geometry'],inplace = True)
df_22['remittances_per_migrant'] = 1_000_000 * df_22['mln_USD_remesas'] / df_22['est_migrants']

df_22 = df_22.merge(gdf[['state', 'geometry']], left_on='state',
              right_on = 'state', how = 'left')

df_22 = geopandas.GeoDataFrame(df_22, geometry='geometry')

##few plots

#violin plot
fig = px.violin(df_22, y="remittances_per_migrant", box=True, # draw box plot inside the violin
                points='all', hover_data=['state'])
fig.update_layout(title = "distribution of remittances per migrant (USD) in 2022")
fig.show()

#map
outfolder = os.getcwd() + "\\mexico\\plots\\remittances_per_state\\"
fig = go.Figure()
fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df_22['geometry'].to_json()),
                                        locations=df_22.index,
                                        z=df_22["remittances_per_migrant"],
                                        text=df_22['state'],
                                        hovertemplate=
                                        '<br>State: %{text}' +
                                        '<br>Remittances received per<br>migrant in the US: %{z:,.0f}',
                                        colorscale="speed", marker_opacity=0.7,
                                        colorbar_title="Remittances per<br>migrant in a year"))
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
fig.update_layout(title=f'Estimated remittances per migrant, by Mexican state')
fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
fig.write_html(outfolder + f"\\remittances_per_state_2022.html")
fig.show()

### all years, see if there is an evolution
df_mig = df_mig.merge(df_rem, right_on = ['state', 'quarter'], left_on = ['mex_state', 'year'])
df_mig.drop(columns = ['mex_state', 'quarter', 'geometry'],inplace = True)
df_mig['remittances_per_migrant'] = 1_000_000 * df_mig['mln_USD_remesas'] / df_mig['nr_registered']

df_mig = df_mig.merge(gdf[['state', 'geometry']], on='state', how = 'left')
df_mig = geopandas.GeoDataFrame(df_mig, geometry='geometry')

#violin
fig = px.violin(df_mig, y="remittances_per_migrant", box=True, # draw box plot inside the violin
                points='all', hover_data=['state'], color = 'year')
fig.update_layout(title = "distribution of remittances per registered migrant (USD)")
fig.update_yaxes(title = "Remittances per registered migrant (USD)")
fig.write_html(os.getcwd() + "\\mexico\\models\\plots\\remittances_per_registered_migrant_violin_overtime.html")
fig.show()

fig = px.violin(df_mig, y="nr_registered", box=True, # draw box plot inside the violin
                points='all', hover_data=['state'], color = 'year')
fig.update_layout(title = "distribution of registered migrants from each state")
fig.update_yaxes(title = "Nr of registered migrants from each state")
fig.write_html(os.getcwd() + "\\mexico\\models\\plots\\registered_migrants_violin_overtime.html")
fig.show()

fig = px.violin(df_mig, y="mln_USD_remesas", box=True, # draw box plot inside the violin
                points='all', hover_data=['state'], color = 'year')
fig.update_layout(title = "distribution of received remittances from each state")
fig.update_yaxes(title = "Mln USD in received remittances")
fig.write_html(os.getcwd() + "\\mexico\\models\\plots\\remesas_recibidas_violin_overtime.html")
fig.show()

###
# expected number of migrants given remittances
df_22['expected_migrants'] = df_22['mln_USD_remesas'] * 1_000_000 / 12 / promedio_usd #assuming one sending of remittances a month
outfolder = os.getcwd() + "\\mexico\\plots\\remittances_per_state\\"
fig = go.Figure()
fig = fig.add_trace(go.Choroplethmapbox(geojson=json.loads(df_22['geometry'].to_json()),
                                        locations=df_22.index,
                                        z=df_22["expected_migrants"],
                                        text=df_22['state'],
                                        hovertemplate=
                                        '<br>State: %{text}' +
                                        '<br>Expected nr migrants<br>given remittances: %{z:,.0f}',
                                        colorscale="speed", marker_opacity=0.7,
                                        colorbar_title="Expected nr migrants<br>given remittances"))
fig.update_layout(hovermode='closest',mapbox=dict(accesstoken=mapbox_access_token,bearing=0,
        center=go.layout.mapbox.Center(lat=23.5,lon=-99),pitch=0,zoom=4.6))
fig.update_layout(title=f'Expected nr migrants given remittances, by Mexican state')
fig.update_geos(fitbounds="locations", lataxis_showgrid=True, lonaxis_showgrid=True, showcountries=True)
fig.write_html(outfolder + f"\\expected_migrants_given_remittances.html")
fig.show()

##scatter
outfolder = 'C:\\git-projects\\csh_remittances\\mexico\\plots\\'
fig = px.scatter(df_22, x="nr_registered", y="expected_migrants", hover_data=['state'],
                 trendline="ols")
fig.update_yaxes(title="nr registered IME")
fig.update_xaxes(title="Expected migrants given remittances")
fig.update_layout(title=f'Migration sources by Mexican state, 2022, CONAPO v. IME')
fig.write_html(outfolder + f"\\IMEvCONAPO\\CONAPO_v_IME_migrants_2022_states.html")
fig.show()
