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
from sklearn.linear_model import LinearRegression
import numpy as np

#load inflow of remittances
df_in = pd.read_excel("C://Data//remittances//inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')

#load outflow of remittances
df_out = pd.read_excel("C://Data//remittances//outward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_out.rename(columns = {"Remittance outflows (US$ million)": "country"}, inplace=True)
df_out = pd.melt(df_out, id_vars=['country'], value_vars=df_out.columns.tolist()[1:])
df_out.rename(columns = {"variable": "year", "value" : "outflow"}, inplace=True)
df_out.year = df_out.year.astype('int')

#merge into one dataframe
df = df_in.merge(df_out, on = ['country', 'year'], how = 'left')

## add the GDP deflator information (Mexico)
gdp_def = pd.read_excel("C://Data//general//gdp_def_all.xls")
gdp_def = pd.melt(gdp_def, id_vars=gdp_def.columns.tolist()[:2], value_vars=gdp_def.columns.tolist()[2:])
gdp_def.rename(columns = {"Country Name":"country", "variable": "year", "value" : "def"}, inplace=True)
gdp_def.year = gdp_def.year.astype('int')

def plot_remittances_flows(country):

    #make a directory to save the plots
    try:
        os.mkdir(os.getcwd() + f"\\plots\\country_flows\\{country}")
    except:
        print("the folder for this country already exists")

    outfolder = os.getcwd() + f"\\plots\\country_flows\\{country}"

    df_def = gdp_def[gdp_def.country == country].ffill()

    df_country = df[df.country == country]
    df_country = df_country.merge(df_def[["country", "year", "def"]], on=["country", "year"], how='left')
    df_country["def"] = df_country["def"] / df_country["def"].iloc[0]
    df_country["inflow_real"] = df_country.inflow / df_country["def"]
    df_country["outflow_real"] = df_country.outflow / df_country["def"]

    ## plot the nominal data for df_countryico
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_country['year'], df_country['inflow'], label='inflow')
    ax.plot(df_country['year'], df_country['outflow'], label='outflow')
    plt.title(f"Nominal inflow and outflow of remittances, {country}, (mln US$)")
    plt.xlabel("year")
    plt.ylabel("mln US$")
    plt.legend()
    plt.grid()
    fig.savefig(outfolder + f"\\{country}_remittances_flows_2000-2023_NOMINAL.png")

    ## plot the real data for df_countryico
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_country['year'], df_country['inflow_real'], label='inflow')
    ax.plot(df_country['year'], df_country['outflow_real'], label='outflow')
    plt.title(f"Real inflow and outflow of remittances, {country}, (mln US$ (2000))")
    plt.xlabel("year")
    plt.ylabel("mln US$ (ind. 2000)")
    plt.legend()
    plt.grid()
    fig.savefig(outfolder + f"\\{country}_remittances_flows_2000-2023_REAL.png")

    ## nominal v real inflow and outflow
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_country['year'], df_country['inflow'], label='nominal inflow')
    ax.plot(df_country['year'], df_country['inflow_real'], label='real inflow')
    plt.title(f"Nominal and real inflow of remittances, {country}, (mln US$)")
    plt.xlabel("year")
    plt.ylabel("mln US$")
    plt.legend()
    plt.grid()
    fig.savefig(outfolder + f"\\{country}_remittances_2000-2023_INFLOWS.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_country['year'], df_country['outflow'], label='nominal outflow')
    ax.plot(df_country['year'], df_country['outflow_real'], label='real outflow')
    plt.title(f"Nominal and real outflow of remittances, {country}, (mln US$)")
    plt.xlabel("year")
    plt.ylabel("mln US$")
    plt.legend()
    plt.grid()
    fig.savefig(outfolder + f"\\{country}_remittances_2000-2023_OUTFLOWS.png")

#plot flows for mexico
plot_remittances_flows("Mexico")

#plot flows for Austria
plot_remittances_flows("Austria")
plot_remittances_flows("Italy")
plot_remittances_flows("Germany")

#
# ##import migration stock data
#
# df_stock = pd.read_excel("C://Data//general//migration_stock_abs.xls")
# df_stock.rename(columns = {"Country Name" : "country"}, inplace = True)
# df_stock = df_stock.drop(columns = "Country Code")
# df_stock = pd.melt(df_stock, id_vars="country", value_vars=df_stock.columns.tolist()[1:])
# df_stock.rename(columns = {"variable" : "year", "value" : "migrant_pop"},inplace = True)
# df_stock.year = df_stock.year.astype(int)
#
# df = df.merge(df_stock, on=["country", "year"], how="left")
# df_pr.rename(columns = {"2015" : "migrant_pop"}, inplace = True)
#
# # regression
# x = df_pr['migrant_pop'].to_numpy().reshape(-1, 1) /1_000_000
# y = df_pr['sent_remittances'].to_numpy() / 1000
# reg = LinearRegression().fit(x, y)
# df_pr['reg'] = reg.predict(x)
#
# ##plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x = df_pr.migrant_pop / 1_000_000,
#                          y = df_pr.sent_remittances / 1000,
#                          mode="markers",
#                          marker = dict(size = 10),
#                          customdata= df_pr["country"],
#                          hovertemplate=
#                          "<b>%{customdata}</b><br>" +
#                          "Migrant population hosted: %{x:.2f}mln people<br>" +
#                          "Remittances sent: %{y:.2f}bn$,<br>" +
#                          # "Life Expectancy: %{y:.0f}<br>"
#                          "<extra></extra>",
#                          showlegend = False
#                          )
#               )
# fig.add_trace(go.Scatter(x = df_pr.migrant_pop / 1_000_000,
#               y = df_pr["reg"],
#               mode = "lines",
#               line = dict(color = "red"),
#               name = "fitted line"
#                          ))
# fig.update_layout(title = "Migrant population hosted vs. remittances sent")
# fig.update_xaxes(title="Total migrant population hosted (mln people, 2015)")
# fig.update_yaxes(title="Sent remittances (bn USD, 2021)")
# fig.write_html("plots//migrant_pop_v_remittances_sent_WITH_USA.html")
# fig.show()
#
#
# df_no_usa = df_pr[df_pr.country != "United States of America"]
# # regression
# x = df_no_usa['migrant_pop'].to_numpy().reshape(-1, 1) /1_000_000
# y = df_no_usa['sent_remittances'].to_numpy() / 1000
# reg = LinearRegression().fit(x, y)
# df_no_usa['reg'] = reg.predict(x)
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(y = df_no_usa.sent_remittances / 1000,
#                          x = df_no_usa.migrant_pop / 1_000_000,
#                          mode="markers",
#                          marker = dict(size = 10),
#                          customdata= df_no_usa["country"],
#                          hovertemplate=
#                          "<b>%{customdata}</b><br>" +
#                          "Migrant population hosted: %{x:.2f}mln people<br>" +
#                          "Remittances sent: %{y:.2f}bn$,<br>" +
#                          # "Life Expectancy: %{y:.0f}<br>"
#                          "<extra></extra>",
#                          showlegend = False
#                          )
#               )
# fig.add_trace(go.Scatter(x = df_no_usa.migrant_pop / 1_000_000,
#               y = df_no_usa["reg"],
#               mode = "lines",
#               line = dict(color = "red"),
#               name = "fitted line"
#                          ))
# fig.update_layout(title = "Migrant population hosted vs. remittances sent (excluding USA)")
# fig.update_xaxes(title="Total migrant population hosted (mln people, 2015)")
# fig.update_yaxes(title="Sent remittances (bn USD, 2021)")
# fig.write_html("plots//migrant_pop_v_remittances_sent_NO_USA.html")
# fig.show()