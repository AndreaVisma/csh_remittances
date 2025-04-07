

import dash
import numpy as np
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go

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

# Initialize Dash app
app = dash.Dash(__name__)

# Get unique values
origins = df['origin'].unique()
destinations = df['destination'].unique()

# App layout
app.layout = html.Div([
    html.H1("Migrant Population Pyramid Explorer"),

    html.Div([
        html.Div([
            html.Label("Origin Country:"),
            dcc.Dropdown(
                id='origin-dropdown',
                options=[{'label': o, 'value': o} for o in origins],
                value=origins[0]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Destination Country:"),
            dcc.Dropdown(
                id='destination-dropdown',
                options=[{'label': d, 'value': d} for d in destinations],
                value=destinations[0]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Select Month:"),
            dcc.Dropdown(
                id='month-dropdown',
                options=date_options,
                value=default_date,
                placeholder="Select Year-Month"
            )
        ], style={'width': '40%', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='population-pyramid')
])


# Callback to update pyramid
@callback(
    Output('population-pyramid', 'figure'),
    [Input('origin-dropdown', 'value'),
     Input('destination-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_pyramid(origin, destination, selected_date):
    # Filter data for the selected date
    filtered = df[
        (df['origin'] == origin) &
        (df['destination'] == destination) &
        (df['date'] == pd.to_datetime(selected_date))
        ]

    # Split into male/female
    male = filtered[filtered['sex'] == 'male'].sort_values('age_group', ascending=True)
    female = filtered[filtered['sex'] == 'female'].sort_values('age_group', ascending=True)

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=male['age_group'],
        x=-male['n_people'],
        name='Male',
        orientation='h',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        y=female['age_group'],
        x=female['n_people'],
        name='Female',
        orientation='h',
        marker_color='red'
    ))

    # Calculate max value for symmetric axis
    max_n = max(filtered['n_people'].max(), abs(filtered['n_people'].min())) if not filtered.empty else 0
    ticks = [int(x) for x in np.linspace(-max_n, max_n, 10)]
    ticks.append(0)
    ticks = np.sort(ticks)

    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        height=800,
        title=f'Population Pyramid: {origin} to {destination} ({pd.to_datetime(selected_date).strftime("%b %Y")})',
        barmode='relative',
        bargap=0.1,
        xaxis=dict(
            title='Number of People',
            range=[-max_n * 1.1, max_n * 1.1] if max_n != 0 else [-10, 10],
            tickvals=ticks,
            ticktext=abs(ticks)
        ),
        yaxis=dict(title='Age Group'),
        hovermode='y unified'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)