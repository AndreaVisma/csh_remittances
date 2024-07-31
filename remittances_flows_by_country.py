"""
Script: remittances_data_download.py
Author: Andrea Vismara
Date: 01/07/2024
Description:
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

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

#load outflow of remittances
df_out = pd.read_excel(data_folder + "outward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y", skiprows=2)
df_out.rename(columns = {"Remittance outflows (US$ million)": "country"}, inplace=True)
df_out = pd.melt(df_out, id_vars=['country'], value_vars=df_out.columns.tolist()[1:])
df_out.rename(columns = {"variable": "year", "value" : "outflow"}, inplace=True)
df_out.replace({"2023e": '2023'}, inplace =True)
df_out.year = df_out.year.astype('int')

## add the GDP deflator information
gdp_def = pd.read_excel(data_folder + "gdp_deflator.xls", skiprows = 3)
gdp_def.drop(columns = ['Country Code', 'Indicator Name', 'Indicator Code'], inplace = True)
gdp_def = pd.melt(gdp_def, id_vars=gdp_def.columns.tolist()[:1], value_vars=gdp_def.columns.tolist()[2:])
gdp_def.rename(columns = {"Country Name":"country", "variable": "year", "value" : "def"}, inplace=True)
gdp_def.year = gdp_def.year.astype('int')

##import migration stock data
df_stock = pd.read_excel(data_folder + "migration_stock_abs.xls", skiprows = 3)
df_stock.drop(columns = ['Country Code', 'Indicator Name', 'Indicator Code'], inplace = True)
df_stock = pd.melt(df_stock, id_vars="Country Name", value_vars=["2000", "2005", "2010", "2015"])
df_stock.rename(columns = {"Country Name":"country", "variable": "year", "value" : "migrants_hosted"}, inplace=True)
df_stock.year = df_stock.year.astype('int')

#merge into one dataframe
df = df_in.merge(df_out, on = ['country', 'year'], how = 'left')
df = df.merge(df_stock, on = ['country', 'year'], how = 'left').merge(
    gdp_def, on = ['country', 'year'], how = 'left')

# check how many missing values we have
print(df.isna().sum())

#########
# Define plotting functions
#########

#incoming and outflowing remittances by country
def plot_remittances_flows(country):

    #make a directory to save the plots
    try:
        os.mkdir(os.getcwd() + f"\\plots\\country_flows\\{country}")
    except:
        print("the folder for this country already exists")

    outfolder = os.getcwd() + f"\\plots\\country_flows\\{country}"

    df_country = df[df.country == country].copy()
    df_country["def"] = df_country["def"].ffill()
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

    print(f"plots saved in {outfolder}")

#shares in the global remittances network
def plot_shares():

    years = df.year.unique().tolist()

    pbar = tqdm(years)

    for year in pbar:
        pbar.set_description(f"plotting {year} data")
        df_year = df[df.year == year]
        #outflows
        fig = px.pie(df_year,
                     values='outflow',
                     names='country',
                     title=f'Remittances outflows {year}')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.write_html(f"plots//shares_total_flows//outflows//outflows_shares_{year}.html")
        #inflows
        fig = px.pie(df_year,
                     values='inflow',
                     names='country',
                     title=f'Remittances inflows {year}')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.write_html(f"plots//shares_total_flows//inflows//inflows_shares_{year}.html")

#comparison of outflow of remittances for different years
def plot_outflows_migrants_by_year(no_USA = False):

    if no_USA:
        df_copy = df[df.country != "United States"]
        label = "NO_USA"
    else:
        df_copy = df.copy()
        label = "WITH_USA"


    years = [2000, 2005, 2010, 2015]
    fig = go.Figure()

    pbar = tqdm(years)
    for year in pbar:
        df_year = df_copy[(df_copy.year == year) &
                          (df_copy.outflow.notna()) &
                          (df_copy.migrants_hosted.notna())].copy()

        # df_year['migrants_hosted'] = df_year['migrants_hosted'].apply(lambda x: np.log(x))
        # df_year['outflow'] = df_year['outflow'].apply(lambda x: np.log(x))
        # regression
        x = df_year['migrants_hosted'].to_numpy().reshape(-1, 1) / 1_000_000
        y = df_year['outflow'].to_numpy() / 1000
        reg = LinearRegression().fit(x, y)
        df_year['reg'] = reg.predict(x)

        ##plot
        fig.add_trace(go.Scatter(x=df_year.migrants_hosted / 1_000_000,
                                 y=df_year.outflow / 1000,
                                 mode="markers",
                                 marker=dict(size=10),
                                 customdata=df_year["country"],
                                 hovertemplate=
                                 "<b>%{customdata}</b><br>" +
                                 "Migrant population hosted: %{x:.2f}mln people<br>" +
                                 "Remittances sent: %{y:.2f}bn$,<br>" +
                                 # "Life Expectancy: %{y:.0f}<br>"
                                 "<extra></extra>",
                                 showlegend=True,
                                 name = year
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_year.migrants_hosted / 1_000_000,
                                 y=df_year["reg"],
                                 mode="lines",
                                 line=dict(color="red"),
                                 name=f"{year}: fitted line"
                                 ))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(title=f"Migrant population hosted vs. remittances sent {label}")
    fig.update_xaxes(title="Total migrant population hosted (mln people, 2015)")
    fig.update_yaxes(title="Sent remittances (bn USD, 2021)")
    fig.write_html(f"plots//remittances_and_migrants//migrants_hosted_remittances_sent_{label}.html")
    fig.show()
    print(f"plot saved in {os.getcwd()}//plots//remittances_and_migrants")

########
# Plot data for selected countries
########

#plot flows for mexico
plot_remittances_flows("Mexico")
plot_remittances_flows("Austria")
plot_remittances_flows("Italy")
plot_remittances_flows("Germany")

#plot shares in total remittances
plot_shares()

#remittances and hosted migrants
plot_outflows_migrants_by_year(no_USA = False)
plot_outflows_migrants_by_year(no_USA = True)

##
import plotly.express as px
fig = px.line(df_out, x = "year", y ="outflow",
                 color = "country", log_x=False, log_y=True,
                 title= "Outflow of remittances by country over time")
fig.show()

fig = px.line(df_in, x = "year", y ="inflow",
                 color = "country", log_x=False, log_y=True,
                 title= "Inflow of remittances by country over time")
fig.show()
