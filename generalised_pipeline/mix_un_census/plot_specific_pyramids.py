


import dash
import numpy as np
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

df = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\complete_stock_to_plot.pkl")
df['date'] = pd.to_datetime(df['date'])
sorter = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
       '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
       '70-74', '75-79', '80-84', '85-89', '90-94', '95-99']
df.age_group = df.age_group.astype("category")
df.age_group = df.age_group.cat.set_categories(sorter)
date_options = [
    {'label': ym, 'value': date}
    for ym, date in zip(df['year_month'].unique(), df['date'].unique())
]
default_date = date_options[0]['value']  # First available date

asy_df = pd.read_pickle("C:\\Data\\migration\\bilateral_stocks\\pyramid_asymmetry_beginning_of_the_year_NEW.pkl")


def plot_population_pyramid(df, origin, destination, year_month):
    """
    Plots a population pyramid for a given origin-destination pair and month.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['date', 'origin', 'age_group', 'sex', 'n_people',
                               'mean_age', 'destination', 'year_month']
    origin : str
        Country of origin to filter on.
    destination : str
        Country of destination to filter on.
    year_month : str
        Period to filter on (e.g. '2015-06').
    """
    # Filter dataframe
    subset = df[(df["origin"] == origin) &
                (df["destination"] == destination) &
                (df["year_month"] == year_month)]

    if subset.empty:
        print("No data available for this selection.")
        return

    # Pivot into age vs sex
    pyramid = subset.pivot_table(index="age_group",
                                 columns="sex",
                                 values="n_people",
                                 aggfunc="sum",
                                 fill_value=0)

    # Ensure both sexes exist
    if "male" not in pyramid.columns:
        pyramid["male"] = 0
    if "female" not in pyramid.columns:
        pyramid["female"] = 0

    # Sort age groups (assumes age_group strings like '0-4','5-9',...)
    try:
        pyramid = pyramid.sort_index(key=lambda x: x.str.extract(r'(\d+)').astype(int))
    except Exception:
        pyramid = pyramid.sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(pyramid.index, -pyramid["male"], label="male", color="steelblue")
    ax.barh(pyramid.index, pyramid["female"], label="female", color="lightcoral")

    ax.set_xlabel("Population")
    ax.set_ylabel("Age Group")
    ax.set_title(f"Population Pyramid\n{origin} → {destination}, {year_month}")
    ax.legend()
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Symmetric x-axis
    max_val = max(pyramid["male"].max(), pyramid["female"].max())
    ax.set_xlim(-max_val * 1.1, max_val * 1.1)

    plt.tight_layout()
    plt.show(block = True)

plot_population_pyramid(df, origin="Botswana", destination="Lesotho", year_month="2015-06")