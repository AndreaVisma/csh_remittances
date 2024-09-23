
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import os
import matplotlib.ticker as mtick
import geopandas
from utils import *
from tqdm import tqdm
import plotly.io as pio
pio.renderers.default = 'browser'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

out_folder = "c:\\git-projects\\csh_remittances\\austria\\plots\\"

##import historical migration data
data_folder = "c:\\data\\migration\\austria\\pop_by_country_origin\\"
df = pd.DataFrame([])
for file_name in tqdm(os.listdir(data_folder)):
    file = data_folder + file_name
    df_small = pd.read_excel(file, skiprows=10, skipfooter=10)
    df_small = df_small.iloc[1:, 1:]
    df_small = df_small.rename(columns = dict(zip(df_small.columns[:2], ['year', 'land'])))
    df_small['year'] = df_small['year'].ffill()

    df = pd.concat([df, df_small])
df = df.sort_values(['year', 'land'])
df['land'] = df.land.apply(lambda x: x.split(' <')[0])

df.to_excel("c:\\data\\migration\\austria\\population_by_nationality_year_land_2010-2024.xlsx", index = False)

########
# now process the remittances info
########
#migrants
df = pd.read_excel("c:\\data\\migration\\austria\\population_by_nationality_year_land_2010-2024.xlsx")
df = df.melt(id_vars = 'year', value_vars=df.columns[2:],
             value_name='pop', var_name='nationality')
df.loc[df['pop'] == '-', 'pop'] = 0
df = df.groupby(['year', 'nationality']).sum()
df = df.reset_index()
df['nationality'] = df['nationality'].map(dict_names)
df = df.dropna()
df.year = df.year.astype(int) - 1
df.rename(columns = {'nationality':'country'}, inplace = True)
#remittances
file = "c:\\data\\remittances\\austria\\remittances_072024.xlsx"
#inflows
df_inflow = pd.read_excel(file, sheet_name="Gast Credit", skiprows=3, skipfooter=4, usecols="B:P")
df_inflow = df_inflow[df_inflow['Herkunftsland'] != 'Total']
df_inflow['Herkunftsland'] = df_inflow['Herkunftsland'].apply(lambda x:
    GoogleTranslator(source='de', target='en').translate(x))
df_inflow = pd.melt(df_inflow, id_vars='Herkunftsland', value_vars=df_inflow.columns[2:],
                    value_name='mln_euros', var_name='year')
df_inflow['Remittances flow'] = 'to Austria'
df_inflow.rename(columns = {'Herkunftsland':'country'}, inplace = True)
#outflows
df_outflow = pd.read_excel(file, sheet_name="Gast Debit", skiprows=3, skipfooter=4, usecols="B:P")
df_outflow = df_outflow[df_outflow['Zielland'] != 'Total']
df_outflow['Zielland'] = df_outflow['Zielland'].apply(lambda x:
    GoogleTranslator(source='de', target='en').translate(x))
df_outflow = pd.melt(df_outflow, id_vars='Zielland', value_vars=df_outflow.columns[2:],
                    value_name='mln_euros', var_name='year')
df_outflow['Remittances flow'] = 'from Austria'
df_outflow.rename(columns = {'Zielland':'country'}, inplace = True)
#merge
df_rem = pd.concat([df_inflow, df_outflow])
df_rem['country'] = df_rem['country'].map(dict_names)

######
# merge everything
df = df.merge(df_rem, on = ['year', 'country'], how = 'outer')
df = df[df.year > 2010]
df.to_excel("c:\\data\\remittances\\austria\\remittances_migrant_pop_austria_2011-2023.xlsx", index = False)

#poisson probability
df = pd.read_excel("c:\\data\\remittances\\austria\\remittances_migrant_pop_austria_2011-2023.xlsx")
for year in tqdm(df.year.unique()):
    try:
        df.loc[df.year == year, 'pct_rem'] = (100 * df.loc[df.year == year, 'mln_euros'] /
        df.loc[df.year == year, 'mln_euros'].sum())
    except:
        print(f"missing remittances for {year}, filling with 0%")
        df.loc[df.year == year, 'pct_rem'] = 0
    try:
        df.loc[df.year == year, 'pct_pop'] = (100 * df.loc[df.year == year, 'pop'] /
        df.loc[df.year == year, 'pop'].sum())
    except:
        print(f"missing population for {year}, filling with 0%")
        df.loc[df.year == year, 'pct_pop'] = 0

df = df[df['pop'] > 0]
df['rem_euros_month'] = df['mln_euros'] * 1_000_000 / 12
df['exp_migrants'] = df['rem_euros_month'] / 400
df["poisson_prob"] = df["exp_migrants"] / df['pop']

### Canada is broken
df = df[df.country != 'Canada']
df = df[df['Remittances flow'] == 'from Austria']

###
countries = [x for x in df.country.unique() if df.loc[df.country == x, 'pct_rem'].mean() > 2]
df_small = df[df.country.isin(countries)]
years = ["2011", "", "","","","","","","","","","","2023"]
fig = go.Figure()
for country in df_small.country.unique():
    df_state = df_small[df_small.country == country]
    fig = fig.add_trace(go.Scatter(x=df_state["pop"], y=df_state["mln_euros"], text=years,
                                   mode="lines+markers+text",
                               marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"), name = country))
fig.update_xaxes(title="Migrant population in Austria")
fig.update_yaxes(title="Million euros in remittances sent from Austria")
fig.update_layout(title=f'Population and remittances sent, <br>by nationality, 2011-2023, pct. remittances > 2%')
fig.write_html(out_folder + f"\\pop_v_rem_2011-2023_more2pct.html")
fig.show()
##
countries = [x for x in df.country.unique() if (0.5 < df.loc[df.country == x, 'pct_rem'].mean()) & (df.loc[df.country == x, 'pct_rem'].mean() < 2)]
df_small = df[df.country.isin(countries)]
years = ["2011", "", "","","","","","","","","","","2023"]
fig = go.Figure()
for country in df_small.country.unique():
    df_state = df_small[df_small.country == country]
    fig = fig.add_trace(go.Scatter(x=df_state["pop"], y=df_state["mln_euros"], text=years,
                                   mode="lines+markers+text",
                               marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"), name = country))
fig.update_xaxes(title="Migrant population in Austria")
fig.update_yaxes(title="Million euros in remittances sent from Austria")
fig.update_layout(title=f'Population and remittances sent, <br>by nationality, 2011-2023, pct. remittances < 2%, >0.5%')
fig.write_html(out_folder + f"\\pop_v_rem_2011-2023_less2pct.html")
fig.show()
## all countries
countries = [x for x in df.country.unique() if (df.loc[df.country == x, 'mln_euros'].mean() > 5) & (df.loc[df.country == x, 'pop'].mean() > 5_000)]
df_small = df[df.country.isin(countries)]
countries_ordered = (df_small[['country', 'pop']].groupby('country').mean().reset_index().
 sort_values('pop', ascending = False)).country.to_list()
fig = go.Figure()
for country in countries_ordered:
    df_state = df_small[df_small.country == country]
    fig = fig.add_trace(go.Scatter(x=df_state["pop"], y=df_state["mln_euros"], text=years,
                                   mode="lines+markers+text",
                               marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"), name = country))
fig.update_xaxes(title="Migrant population in Austria")
fig.update_yaxes(title="Million euros in remittances sent from Austria")
fig.update_layout(legend_title_text='Countries ordered<br>by average<br>migrant population')
fig.update_layout(title=f'Population and remittances sent, by nationality,<br>2011-2023, mean remittances > 5 mln, mean population > 5k')
fig.write_html(out_folder + f"\\pop_v_rem_2011-2023_all.html")
fig.show()
###

######
# final iteration : present only interesting cases
#####
countries = ['Afghanistan', 'Czechia', 'Germany', 'Syria', 'Ukraine', 'Serbia', 'Hungary', 'Turkey', 'Croatia', 'Romania']
years = ["2011", "", "","","","","","","","","","","2023"]
df_small = df[df.country.isin(countries)]
countries_ordered = (df_small[['country', 'pop']].groupby('country').mean().reset_index().
 sort_values('pop', ascending = False)).country.to_list()
fig = go.Figure()
for country in countries_ordered:
    df_state = df_small[df_small.country == country]
    fig = fig.add_trace(go.Scatter(x=df_state["pop"], y=df_state["mln_euros"], text=years,
                                   mode="lines+markers+text",
                               marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"), name = country))
fig.update_xaxes(title="Migrant population in Austria")
fig.update_yaxes(title="Million euros in remittances sent from Austria")
fig.update_layout(legend_title_text='Countries ordered<br>by average<br>migrant population')
fig.update_layout(title=f'Diaspora population and remittances sent, by nationality,<br>2011-2023')
fig.update_layout(
    autosize=False,
    width=1400,
    height=800)
fig.write_html(out_folder + f"\\BMI_praesi_countries.html")
# fig.write_image(out_folder + f"\\BMI_praesi_countries.svg")
fig.show()
#############

##population by country
fig = px.line(df, 'year', 'pop', color = 'country')
fig.update_yaxes(title = 'Diaspora numbers in Austria per year')
fig.show()

##remittances by country
fig = px.line(df, 'year', 'mln_euros', color = 'country')
fig.update_yaxes(title = 'Remittances sent from Austria per year')
fig.show()

##poisson probability
fig = px.line(df, 'year', 'poisson_prob', color = 'country')
fig.update_yaxes(title = 'Probability of each diaspora in Austria sending money per year')
fig.show()

##calculate percentage change in probability of sending remittances
for country in tqdm(df.country.unique()):
    df.loc[df.country == country, 'pct_change_poisson_prob'] = (
            df.loc[df.country == country, 'poisson_prob'].pct_change() * 100)

##change in poisson probability
fig = px.line(df, 'year', 'pct_change_poisson_prob', color = 'country')
fig.update_yaxes(title = 'Percentage change in the probability<br> of each diaspora in Austria sending money per year')
fig.show()