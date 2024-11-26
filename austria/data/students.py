import pandas as pd
from utils import dict_names
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

df_stud = pd.read_excel("c:\\data\\population\\austria\\students_by_origin.xlsx",
                        skiprows=10, usecols = "B:D")
df_stud.ffill(inplace = True)
df_stud['year'] = df_stud.semester.apply(lambda x: '20' + x[-5:-3]).astype(int)
df_stud.drop(columns = 'semester', inplace = True)
df_stud['quarter'] = 3
df_stud.country = df_stud.country.map(dict_names)
df_stud = df_stud.replace('-', np.nan)
df_stud.dropna(inplace = True)

### remittances
df_rem = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df_rem = df_rem.merge(df_stud, on = ['country', 'year', 'quarter'], how = 'left')
for country in tqdm(df_rem.country.unique()):
    df_rem.loc[df_rem.country == country, 'students'] = df_rem.loc[df_rem.country == country, 'students'].ffill()
df_rem['students'].fillna(0, inplace = True)
df_rem['pct_students'] = 100 * df_rem['students'] / df_rem['population']
df_rem['pct_students'] = df_rem['pct_students'].clip(0,100)

df_rem.to_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx", index = False)


countries = ['Italy', 'Germany', 'Switzerland', 'Greece', 'Spain', 'Belgium']
fig = px.line(df_rem[(df_rem.quarter == 3) & (df_rem.country.isin(countries))], x = 'year', y = 'pct_students', color = 'country')
fig.update_layout(title = 'Percentage of migrant population which is studying in university',
                  plot_bgcolor='white')
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
fig.layout.yaxis.ticksuffix = '%'
fig.show()

country = 'Turkey'
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_rem[(df_rem.quarter == 3) & (df_rem.country == country)]['year'],
                         y = df_rem[(df_rem.quarter == 3) & (df_rem.country == country)]['students'], name = 'Students'))
fig.add_trace(go.Scatter(x = df_rem[(df_rem.quarter == 3) & (df_rem.country == country)]['year'],
                         y = df_rem[(df_rem.quarter == 3) & (df_rem.country == country)]['population'], name = 'All migrants'))
fig.update_layout(title = f'Number of migrants (of which students) from {country} living in Austria',
                  plot_bgcolor='white')
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
fig.show()
