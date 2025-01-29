import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'

df_rem = pd.read_csv('c:\\data\\remittances\\italy\\monthly_splined_remittances.csv')
df_rem["date"] = pd.to_datetime(df_rem["date"])

df_conf = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_conflicts.csv")
df_conf['start_week'] = pd.to_datetime(df_conf['start_week'])
df_conf_monthly = (
    df_conf.groupby(['country', pd.Grouper(key='start_week', freq='M')])
    .agg({'deaths': 'sum'})
    .reset_index()
    .rename(columns={'start_week': 'date'})
)

# Merge with remittances data (left join to keep all remittance months)
df_merged = pd.merge(
    df_rem,
    df_conf_monthly,
    on=['country', 'date'],
    how='left'
)
df_merged.deaths.fillna(0, inplace = True)
df_merged['date'] = pd.to_datetime(df_merged['date'])

# Build the dashboard
# -------------------
fig = make_subplots(specs=[[{"secondary_y": True}]])
country_trace_indices = {}  # Track trace indices per country

# Add traces for each country
for country in df_merged['country'].unique():
    country_data = df_merged[df_merged['country'] == country]
    start_idx = len(fig.data)  # Track starting index

    # 1. Add remittances line (primary axis)
    fig.add_trace(
        go.Scatter(
            x=country_data['date'],
            y=country_data['remittances'],
            name=f'{country} Remittances',
            line=dict(color='blue'),
            visible=False
        ),
        secondary_y=False
    )

    # 2. Add deaths bars (secondary axis)
    fig.add_trace(
        go.Scatter(
            x=country_data['date'],
            y=country_data['deaths'],
            name=f'{country} Conflict Deaths',
            marker=dict(color='red', opacity=0.6),
            hovertext=country_data.apply(
                lambda row: f"Deaths: {row['deaths']:,.0f}",
                axis=1
            ),
            visible=False
        ),
        secondary_y=True
    )

    # Store trace indices for this country
    country_trace_indices[country] = {
        'start': start_idx,
        'end': len(fig.data)  # After adding both traces
    }

# Configure dropdown to toggle country visibility
buttons = []
for country in df_merged['country'].unique():
    trace_indices = range(
        country_trace_indices[country]['start'],
        country_trace_indices[country]['end']
    )
    visible = [False] * len(fig.data)
    for idx in trace_indices:
        visible[idx] = True

    buttons.append(
        dict(
            label=country,
            method='update',
            args=[
                {"visible": visible},
                {"title": f"Remittances & Conflict Deaths: {country}"}
            ]
        )
    )

# Set initial visibility for the first country
first_country = df_merged['country'].unique()[0]
for idx in range(country_trace_indices[first_country]['start'], country_trace_indices[first_country]['end']):
    fig.data[idx].visible = True

# Update layout for clarity
fig.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "showactive": True,
        "x": 0.1,
        "y": 1.15
    }],
    title="Remittances from Italy vs. Conflict Deaths by Country",
    yaxis_title="Remittances (EUR)",
    yaxis2_title="Conflict Deaths",
    hovermode="x unified",
    legend_title="Metric"
)

# Adjust bar width for better visibility
fig.update_traces(marker=dict(line=dict(width=0.5)))
fig.show()