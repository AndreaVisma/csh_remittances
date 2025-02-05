import pandas as pd
import pycountry

def country_to_alpha3(country_name):
    try:
        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except LookupError:
        return None  # Handle missing/custom names

df = pd.read_csv("C:\\Data\\remittances\\italy\\monthly_splined_remittances.csv")
df = df[['country', 'remittances']].groupby('country').mean().reset_index()
df["ISO"] = df["country"].apply(country_to_alpha3)
df['remittances'] = df['remittances'].map(np.log)

import plotly.express as px

fig = px.choropleth(
    df,
    locations="ISO",          # ISO Alpha-3 codes
    color="remittances",      # Values to color
    hover_name="country",     # Show country name on hover
    color_continuous_scale=px.colors.sequential.Purples,
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