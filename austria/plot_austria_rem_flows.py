import pandas as pd

import networkx as nx
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib as mpl
mpl.use("Qtagg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = "browser"

file = "c:\\data\\remittances\\austria\\remittances_072024.xlsx"
outfolder = os.getcwd() + "\\austria\\plots\\"

#inflows
df_inflow = pd.read_excel(file, sheet_name="Gast Credit", skiprows=3, skipfooter=4, usecols="B:P")
df_inflow = pd.melt(df_inflow, id_vars='Herkunftsland', value_vars=df_inflow.columns[2:],
                    value_name='mln_euros', var_name='year')
df_inflow['Remittances flow'] = 'to Austria'
df_inflow.rename(columns = {'Herkunftsland':'country'}, inplace = True)

#outflows
df_outflow = pd.read_excel(file, sheet_name="Gast Debit", skiprows=3, skipfooter=4, usecols="B:P")
df_outflow = pd.melt(df_outflow, id_vars='Zielland', value_vars=df_outflow.columns[2:],
                    value_name='mln_euros', var_name='year')
df_outflow['Remittances flow'] = 'from Austria'
df_outflow.rename(columns = {'Zielland':'country'}, inplace = True)

out_group = df_outflow[['year', 'mln_euros']].groupby('year').sum().reset_index()
out_group = out_group[(out_group.year > 2010) & (out_group.year < 2021)]
print(f"Total remittances sent between 2011 and 2023: {1.12 * out_group.mln_euros.sum() / 1_000} billion euros")
print(f"disasters response = {0.011 * 1.12 * out_group.mln_euros.sum()} million euros")

#merge
df = pd.concat([df_inflow, df_outflow])

## total flows
tot = df[df.country == "Total"]
tot["mln_euros"] = tot["mln_euros"] / 1000
fig = px.bar(tot, x="year", y="mln_euros", color = "Remittances flow",
              title='Remittances flows to and from Austria, 2013-2023')
fig.update_yaxes(title = "<b>Billion (Milliarden) EUR in remittances</b>")
fig.update_xaxes(title = "<b>Year</b>")
fig.write_html(outfolder + "total_remittances_flows_mld_bar.html")
fig.write_image(outfolder + "total_remittances_flows_mld_bar.png")
fig.show()

## flows to each country
def plot_flows_country(country, show = False):

    df_country = df[df.country == country].copy()
    fig = px.line(df_country, x="year", y="mln_euros", color="Remittances flow",
                  title=f'Remittances flows between Austria and {country}, 2013-2023')
    fig.update_yaxes(title="Million EUR in remittances")
    fig.write_html(outfolder + f"country_flows\\{country}_remittances_flows.html")
    if show:
        fig.show()

all_countries = df.country.unique()
for country in tqdm(all_countries):
    plot_flows_country(country, show = False)

##piecharts per year
def plot_pie_year(year, show = False):
    outfolder = os.getcwd() + "\\austria\\plots\\piecharts\\"
    df_year = df[(df.year == year) & (df['Remittances flow'] == 'from Austria') &
                 (df.country != "Total")].copy()
    fig1 = px.pie(df_year, values="mln_euros",
                 names='country',
                 title=f'Remittances flow from Austria<br>to countries in {year}')
                 # color_discrete_sequence=px.colors.sequential.speed)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    fig1.write_html(outfolder + f"piechart_outflow_{year}.html")
    df_year = df[(df.year == year) & (df['Remittances flow'] == 'to Austria') &
                 (df.country != "Total")].copy()
    fig2 = px.pie(df_year, values="mln_euros",
                 names='country',
                 title=f'Remittances flow to Austria<br>from countries in {year}')
                 # color_discrete_sequence=px.colors.sequential.speed)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.write_html(outfolder + f"piechart_inflow_{year}.html")
    if show:
        fig1.show(), fig2.show()

all_years = df.year.unique()
for year in tqdm(all_years):
    plot_pie_year(year, show = False)

## top 10 outflows over time
top10_countries_out = (df[(df.year == 2023) & (df["Remittances flow"] == "from Austria") & (df.country != "Total")].
                       sort_values('mln_euros', ascending = False)).head(10).country.to_list()
df_10_out = df[df.country.isin(top10_countries_out) & (df["Remittances flow"] == "from Austria")]
df_10_out['share'] = 0
for year in df_10_out.year.unique():
    df_10_out.loc[df_10_out.year == year, 'share'] = 100 * df_10_out.loc[df_10_out.year == year, 'mln_euros'] / df_10_out.loc[df_10_out.year == year, 'mln_euros'].sum()

fig = px.bar(df_10_out, x="year", y="share", color="country", hover_data= ['mln_euros'],
             title="Outflows of remittances from Austria, shares (top 10 receivers in 2023)")
fig.write_html(outfolder + f"barchart_outflows_top10.html")
fig.show()

top10_countries_in = (df[(df.year == 2023) & (df["Remittances flow"] == "to Austria") & (df.country != "Total")].
                       sort_values('mln_euros', ascending = False)).head(10).country.to_list()
df_10_in = df[df.country.isin(top10_countries_in) & (df["Remittances flow"] == "to Austria")]
df_10_in['share'] = 0
for year in df_10_in.year.unique():
    df_10_in.loc[df_10_in.year == year, 'share'] = 100 * df_10_in.loc[df_10_in.year == year, 'mln_euros'] / df_10_in.loc[df_10_in.year == year, 'mln_euros'].sum()

fig = px.bar(df_10_in, x="year", y="share", color="country", hover_data= ['mln_euros'],
             title="Outflows of remittances from Austria, shares (top 10 receivers in 2023)")
fig.write_html(outfolder + f"barchart_inflows_top10.html")
fig.show()