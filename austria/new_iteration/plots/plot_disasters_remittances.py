import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

df = pd.read_excel("C:\\Data\\natural_disasters\\disasters_start_end_dates.xlsx")
df = df[df.total_affected > 0]
df['log_affected'] = df.total_affected.map(np.log)
df_rem = pd.read_csv("c:\\data\\my_datasets\\weekly_remittances\\weekly_remittances_austria.csv")
df_rem['remittances'] /= 52

# Ensure start_date and end_date are in datetime format
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df_rem['date'] = pd.to_datetime(df_rem['date'])

def plot_disasters_country(country):
    df_country = df[df.country == country]
    df_country = df_country[df_country['start_date'] >= "01-2012"]
    df_country.sort_values('total_affected', ascending = True, inplace = True)
    df_rem_country = df_rem[df_rem.country == country]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    total_affected = df_country['log_affected']
    normalized_affected = (total_affected - total_affected.min()) / (total_affected.max() - total_affected.min())

    # Define a color scale (e.g., Blues)
    color_scale = px.colors.sequential.YlOrBr

    fig.add_trace(go.Scatter(
        x=df_rem_country['date'],
        y=df_rem_country['remittances'],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Remittances',
        hovertemplate='Date: %{x}<br>Remittances: %{y}',
        yaxis='y2'  # Assign to secondary y-axis
    ))

    # Add rectangles for disaster durations
    for index, row in df_country.iterrows():
        # Get the color for this rectangle based on normalized 'total_affected'
        color_index = int(normalized_affected[index] * (len(color_scale) - 1))  # Map to color scale index
        color = color_scale[color_index]  # Get the corresponding color

        fig.add_trace(go.Scatter(
            x=[row['start_date'], row['end_date'], row['end_date'], row['start_date'], row['start_date']],
            # Rectangle x-coordinates
            y=[0,2,2,0,0],
            # Rectangle y-coordinates
            mode='lines',
            fill='toself',  # Fill the area to create a rectangle
            fillcolor=color,  # Light blue fill color,
            name = row['total_affected'],
            line=dict(color=color, width=2),  # Rectangle border
            hoverinfo='text',
            text=f"Country: {row['country']}<br>Start: {row['start_date']}<br>End: {row['end_date']}<br>Affected: {row['total_affected']}"),
            secondary_y = False
        )
    # Update layout for better readability
    fig.update_yaxes(title = "Remittances")  # Sort countries by total affected
    fig.update_layout(
        title=f'Duration and Intensity of Natural Disasters in {country}',
        xaxis_title='Time',
        yaxis_title='Disasters',
        coloraxis_colorbar_title='Intensity (People Affected)'
    )

    fig.show()

plot_disasters_country('Mexico')
plot_disasters_country('Turkey')
plot_disasters_country('China')
plot_disasters_country('Germany')
plot_disasters_country('Syria')
plot_disasters_country('Afghanistan')