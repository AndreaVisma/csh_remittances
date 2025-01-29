import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'

df_rem = pd.read_csv('c:\\data\\remittances\\italy\\monthly_splined_remittances.csv')
df_rem["date"] = pd.to_datetime(df_rem["date"])
df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year

df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]

# Preprocess data to align disasters with remittance months
# ---------------------------------------------------------
# Convert week_start to monthly frequency and aggregate by disaster type
df_nat_monthly = (
    df_nat.groupby(['country', 'type', pd.Grouper(key='week_start', freq='M')])
    .agg({'total_affected': 'sum', 'total_damage': 'sum'})
    .reset_index()
    .rename(columns={'week_start': 'date'})
)

# Merge with remittances data (left join to keep all remittance months)
df_merged = pd.merge(
    df_rem,
    df_nat_monthly,
    on=['country', 'date'],
    how='left'
)
df_merged['type'] = df_merged['type'].fillna('No Disaster')

# Aggregate total_affected by country, date, and disaster type
df_disaster = (
    df_merged.groupby(['country', 'date', 'type'])
    .agg({'total_affected': 'sum', 'total_damage': 'sum'})
    .reset_index()
)

# Create color mapping for disaster types
disaster_types = df_disaster['type'].unique()
colors = px.colors.qualitative.Plotly
color_map = {dtype: colors[i % len(colors)] for i, dtype in enumerate(disaster_types)}

# Build the dashboard with bar charts
# ------------------------------------
fig = make_subplots(specs=[[{"secondary_y": True}]])
country_trace_indices = {}  # Track trace indices per country

for country in df_disaster['country'].unique():
    country_data = df_merged[df_merged['country'] == country]
    disaster_data = df_disaster[df_disaster['country'] == country]
    start_idx = len(fig.data)  # Track starting index

    # 1. Add remittances line (primary axis)
    fig.add_trace(
        go.Scatter(
            x=country_data['date'],
            y=country_data['remittances'],
            name=f'{country} Remittances',
            line=dict(color='black'),
            visible=False
        ),
        secondary_y=False
    )

    # 2. Add stacked bars for disasters (secondary axis)
    # Get unique disaster types for this country
    types = disaster_data['type'].unique()
    for dtype in types:
        if dtype == 'No Disaster':
            continue
        df_dtype = disaster_data[disaster_data['type'] == dtype]
        fig.add_trace(
            go.Bar(
                x=df_dtype['date'],
                y=df_dtype['total_affected'],
                name=dtype,
                marker=dict(color=color_map[dtype]),
                hovertext=df_dtype.apply(
                    lambda row: f"Affected: {row['total_affected']:,.0f}<br>Damage: ${row['total_damage']:,.0f}",
                    axis=1
                ),
                visible=False,
                showlegend=True
            ),
            secondary_y=True
        )

    # Store trace indices for this country
    country_trace_indices[country] = {
        'start': start_idx,
        'end': len(fig.data)  # After adding all traces
    }

# Configure dropdown to toggle country visibility
buttons = []
for country in df_disaster['country'].unique():
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
                {"title": f"Remittances from Italy & Disasters: {country}"}
            ]
        )
    )

# Set initial visibility for the first country
first_country = df_disaster['country'].unique()[0]
for idx in range(country_trace_indices[first_country]['start'], country_trace_indices[first_country]['end']):
    fig.data[idx].visible = True

# Update layout for stacked bars and styling
fig.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "showactive": True,
        "x": 0.1,
        "y": 1.15
    }],
    barmode='stack',  # Critical for stacking disaster bars
    title="Remittances from Italy vs. Disasters (Stacked by Type)",
    yaxis_title="Remittances (EUR)",
    yaxis2_title="Percentage of population affected (%)",
    hovermode="x unified",
    legend_title="Disaster Type"
)

fig.show()