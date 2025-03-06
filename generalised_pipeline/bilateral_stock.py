
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
from utils import dict_names

un_file = "C:\\Data\\migration\\bilateral_stocks\\un_stock_by_sex_destination_and_origin.xlsx"

df = pd.read_excel(un_file, sheet_name="Table 1", skiprows=10)

df.rename(columns={'Region, development group, country or area of origin' : 'origin',
                   'Region, development group, country or area of destination' : 'destination'}, inplace = True)

df['origin'] = df['origin'].str.replace('*', '')
df = df[df.origin.isin(dict_names.keys())]
df['origin'] = df.origin.map(dict_names)

df['destination'] = df['destination'].str.replace('*', '')
df = df[df.destination.isin(dict_names.keys())]
df['destination'] = df.destination.map(dict_names)

def plot_share_emigrants_by_destination(country_emigration, year):

    df_country = df[df.origin == country_emigration]
    df_country = df_country[["destination", year]]
    df_country['pct'] = 100 * df_country[year] / df_country[year].sum()
    df_country.sort_values('pct', ascending=False, inplace = True)
    df_country = df_country.iloc[:10]

    print(df_country[['destination', 'pct']].head())

    # Define source and target indices
    source = [0] * len(df_country)  # Assume all migrants come from "Origin" (index 0)
    target = list(range(1, len(df_country) + 1))  # Destination indices

    # Create labels (Origin + Destinations)
    labels = [country_emigration] + list(df_country['destination'])

    # Define flow values
    values = list(df_country['pct'])

    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            customdata=values,  # Attach data to show on hover
            hovertemplate='Percentage of migrants: %{customdata}%'
        )
    ))

    fig.update_layout(title_text=f"Residence distribution for migrants from {country_emigration} in {year}, top 10 countries", font_size=12)
    fig.show()


plot_share_emigrants_by_destination("Mexico", 2015)