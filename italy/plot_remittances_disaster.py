import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'

df_rem = pd.read_csv('c:\\data\\remittances\\italy\\monthly_splined_remittances.csv')
df_rem["date"] = pd.to_datetime(df_rem["date"])

df_nat = pd.read_csv("C:\\Data\\my_datasets\\weekly_disasters.csv")
df_nat["week_start"] = pd.to_datetime(df_nat["week_start"])
df_nat["year"] = df_nat.week_start.dt.year

df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
df_nat = df_nat.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df_nat['total_affected'] = 100 * df_nat['total_affected'] / df_nat["population"]
df_nat = df_nat[["week_start", "total_affected", "total_damage", "country", "type"]]

# Aggregate disaster data to monthly level (summing distributed values)
df_nat_monthly = (df_nat.groupby(['country', pd.Grouper(key='week_start', freq='M')])
    .agg({'total_affected': 'sum', 'total_damage': 'sum', 'type': 'first'}).reset_index()
    .rename(columns={'week_start': 'date'}))

df_merged = pd.merge(df_rem,df_nat_monthly,on=['country', 'date'],how='left')
df_merged.fillna(0, inplace = True)

## Time series overlay plot
def plot_country(country):
    df_country = df_merged[df_merged['country'] == country].sort_values('date')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_country['date'],y=df_country['remittances'],
            name='Remittances', line=dict(color='blue')),secondary_y=False)
    fig.add_trace(
        go.Bar(
            x=df_country['date'],
            y=df_country['total_affected'],
            name='Percentage of population affected (%)',
            marker=dict(color='red', opacity=0.4),
            hoverinfo='text',
            hovertext=df_country['type']  # Show disaster type on hover
        ),
        secondary_y=True
    )
    fig.update_layout(
        title=f"Remittances vs. Natural Disasters: {country}",
        xaxis_title='Date',
        yaxis_title='Remittances (euros)',
        yaxis2_title='Percentage of population affected (%)',
        hovermode='x unified')
    fig.show()
    fig.to_html(f"C:\\git-projects\\csh_remittances\\italy\\plots\\remittances_disasters\\{country}.html")

# Example call
plot_country('Philippines')
plot_country('Bangladesh')
plot_country('Pakistan')
