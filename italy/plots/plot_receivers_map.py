import pandas as pd
import pycountry
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

def country_to_alpha3(country_name):
    try:
        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except LookupError:
        return np.nan  # Handle missing/custom names

df = pd.read_csv("C:\\Data\\remittances\\italy\\monthly_splined_remittances.csv")
df['date'] = pd.to_datetime(df.date)
df = df[df.date.dt.year > 2018]
df = df[['country', 'remittances']].groupby('country').mean().reset_index()
df["ISO"] = df["country"].apply(country_to_alpha3)
df.loc[df.country == "Turkey", 'ISO'] = "TUR"
df.loc[df.country == "Congo, Dem. Rep.", 'ISO'] = "COD"
df.loc[df.country == "CAR", 'ISO'] = "CAF"
df.loc[df.country == "Niger", 'ISO'] = "NER"
df['remittances'] = df['remittances'].map(np.log)
df = df[df.remittances > 0]

# plot figure
fig = px.choropleth(
    df,
    locations="ISO",          # ISO Alpha-3 codes
    color="remittances",      # Values to color
    hover_name="country",     # Show country name on hover
    color_continuous_scale=px.colors.sequential.Blues,
    title="Mean monthly remittance flow from Italy",
    labels={"Remittances": "Remittances (euros)"}
)
fig.update_layout(
    geo=dict(
        showframe=False,      # Hide map frame
        showcoastlines=False, # Simplify borders
        projection_type="natural earth"  # Global projection
    )
)
fig.show()