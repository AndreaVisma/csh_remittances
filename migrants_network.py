"""
Script: migrants_network.py
Author: Andrea Vismara
Date: 04/07/2024
Description:
"""

#import
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import re
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

### globals and utils
from utils import *

###########
# load migrants flows data
##########
data_folder = os.getcwd() + "\\data_downloads\\data\\"

#all data
df_all = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
                       skiprows=10, usecols="B,F,H:N", sheet_name="Table 1")
df_all.rename(columns={'Region, development group, country or area of destination': 'destination',
                   'Region, development group, country or area of origin': 'origin'}, inplace=True)
#data for males
# df_males = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
#                          skiprows=11, usecols="B,F,O:U", sheet_name="Table 1", header = None)
# df_males.columns = df_all.columns
#
# #data for females
# df_females = pd.read_excel(data_folder + "undesa_pd_2020_ims_stock_by_sex_destination_and_origin.xlsx",
#                            skiprows=11, usecols="B,F,V:AB", sheet_name="Table 1", header = None)
# df_females.columns = df_all.columns

df_all['origin'] = df_all['origin'].apply(lambda x: remove_asterisc(x))
df_all['origin'] = clean_country_series(df_all['origin'])
df_all['destination'] = df_all['destination'].apply(lambda x: remove_asterisc(x))
df_all['destination'] = clean_country_series(df_all['destination'])
df_all.dropna(inplace = True)

###########
## print some stats
###########

years = df_all.columns[2:].tolist()
print("Global migrants in")
for year in years:
    print(f"""{year}: {round(df_all[year].sum() / 1e6, 2)} mln""")

print("================================")
print("Migrants in Austria in")
for year in years:
    print(f"""{year}: {round(df_all[df_all.destination == "Austria"][year].sum() / 1e6, 2)} mln""")
###########
# define plotting functions
##########

def inflows_outflows_country_sankey(country, year, group = 'all', show = False):

    try:
        os.mkdir(os.getcwd() + f"\\plots\\migrants_stock\\sankey_charts\\{country}")
        out = os.getcwd() + f"\\plots\\migrants_stock\\sankey_charts\\{country}\\"
    except:
        out = os.getcwd() + f"\\plots\\migrants_stock\\sankey_charts\\{country}\\"

    df_year = df_all[["origin", "destination", year]].copy()
    df_year.sort_values(year, ascending = False, inplace = True)

    #country as origin
    df_or = df_year[df_year.origin == country].copy()
    labels = df_or.destination.to_list()
    labels.append(country)
    source = [labels.index(country)] * len(df_or)
    target = [x for x in range(len(df_or))]
    value = df_or[year].to_list()

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

    fig.update_layout(title_text=f"Distribution of migrants from {country} in {year}", font_size=10)
    fig.write_html(out + f"out_stock_{year}_{group}.html")
    if show:
        fig.show()

    # country as destination
    df_dest = df_year[df_year.destination == country].copy()
    labels = df_dest.origin.to_list()
    labels.append(country)
    target = [labels.index(country)] * len(df_dest)
    source = [x for x in range(len(df_dest))]
    value = df_dest[year].to_list()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="red"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text=f"Distribution of migrants in {country} in {year}", font_size=10)
    fig.write_html(out + f"in_stock_{year}_{group}.html")
    if show:
        fig.show()

    #######
    # pie charts
    #######

    try:
        os.mkdir(os.getcwd() + f"\\plots\\migrants_stock\\pie_charts\\{country}")
        out = os.getcwd() + f"\\plots\\migrants_stock\\pie_charts\\{country}\\"
    except:
        out = os.getcwd() + f"\\plots\\migrants_stock\\pie_charts\\{country}\\"

    #origin
    df_or["share"] = 100 * df_or[year] / df_or[year].sum()
    df_or.loc[df_or['share'] < 1, 'destination'] = 'Other countries'  # Represent only large countries
    df_or.rename(columns = {year : "migrants"}, inplace = True)
    fig = px.pie(df_or, values="migrants", names='destination', title=f"Distribution of migrants from {country} in {year}")
    fig.write_html(out + f"out_stock_{year}_{group}.html")
    if show:
        fig.show()

    #destination
    df_dest["share"] = 100 * df_dest[year] / df_dest[year].sum()
    df_dest.loc[df_dest['share'] < 1, 'origin'] = 'Other countries'  # Represent only large countries
    df_dest.rename(columns = {year : "migrants"}, inplace = True)
    fig = px.pie(df_dest, values="migrants", names='origin', title=f"Distribution of migrants in {country} in {year}")
    fig.write_html(out + f"in_stock_{year}_{group}.html")
    if show:
        fig.show()

for country in tqdm(df_all.origin.unique(), total = len(df_all.origin.unique()), position=0, leave=True, colour='green', ncols = 80):
    # for year in tqdm(df_all.columns[2:].tolist(), position=1, desc="years", leave=True, colour='red', ncols=80):
    for year in df_all.columns[2:].tolist():
        inflows_outflows_country_sankey(country, year, group='all')


