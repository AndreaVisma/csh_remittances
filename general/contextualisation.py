
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import os
from utils import dict_names


# define global variables
data_folder = os.getcwd() + "\\data_downloads\\data\\"

######
# Load all the data in
######

#load inflow of remittances
df_in = pd.read_excel(data_folder + "inward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y")
df_in.rename(columns = {"Remittance inflows (US$ million)": "country"}, inplace=True)
df_in = pd.melt(df_in, id_vars=['country'], value_vars=df_in.columns.tolist()[1:])
df_in.rename(columns = {"variable": "year", "value" : "inflow"}, inplace=True)
df_in.year = df_in.year.astype('int')
df_in = df_in[df_in.year == 2023]
df_in['inflow'] *= 1_000_000

#load outflow of remittances
df_out = pd.read_excel(data_folder + "outward-remittance-flows-2024.xlsx",
                      nrows = 214, usecols="A:Y", skiprows=2)
df_out.rename(columns = {"Remittance outflows (US$ million)": "country"}, inplace=True)
df_out = pd.melt(df_out, id_vars=['country'], value_vars=df_out.columns.tolist()[1:])
df_out.rename(columns = {"variable": "year", "value" : "outflow"}, inplace=True)
df_out.replace({"2023e": '2023'}, inplace =True)
df_out.year = df_out.year.astype('int')
df_out = df_out[df_out.year == 2023]
df_out['outflow'] *= 1_000_000

##### merge
df_rem = df_in.merge(df_out, on = ['country', 'year'])
print(set(df_rem.country) - set(dict_names.keys()))
df_rem['country'] = df_rem.country.map(dict_names)


########GDP
df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_clean.xlsx")
df_gdp = df_gdp[df_gdp.year == 2023]
print(set(df_gdp.country) - set(dict_names.keys()))
df_gdp['country'] = df_gdp.country.map(dict_names)

df_rem = df_rem.merge(df_gdp, on = ['country', 'year'])
df_rem['pct_in'] = 100 * df_rem['inflow'] / df_rem['gdp']
df_rem['pct_out'] = 100 * df_rem['outflow'] / df_rem['gdp']
df_rem = df_rem[df_rem.gdp > 1e10]

### regional classification
income_class = pd.read_excel("C:\\Data\\economic\\income_classification_countries_wb.xlsx")
income_class['country'] = income_class.country.map(dict_names)

df_rem = df_rem.merge(income_class[['country', 'group', 'Region']], on = 'country', how = 'left')

##### plot with plotly express
# Create scatter plot
fig = px.scatter(
    df_rem,
    x='pct_in',
    y='pct_out',
    labels={'pct_in': 'Inflow', 'pct_out': 'Outflow', 'country' : 'Country'},
    title='Inflow vs outflow of remittances as percentage of GDP by country (2023)',
    color = 'country',
)
fig.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df_rem,
    x='pct_in',
    y='pct_out',
    hue='Region',
    s=100  # Size of points
)

plt.title('Inflow vs Outflow of Remittances as % of GDP by Country (2023)', fontsize=14)
plt.xlabel('Inflow (% of GDP)')
plt.ylabel('Outflow (% of GDP)')
plt.legend(title='Region')
plt.grid(True)
plt.tight_layout()
plt.show(block = True)

#### Boxplot
melted_df = df_rem.melt(id_vars='country', value_vars=['pct_in', 'pct_out'], var_name='type', value_name='percentage')
sns.boxplot(x='type', y='percentage', data=melted_df)
plt.title('Boxplot of Remittance Inflow and Outflow Percentages')
plt.ylabel('Percentage of GDP')
plt.xlabel('Remittance Type')
plt.tight_layout()
plt.grid(True)
plt.show(block = True)

### barcharts
top_inflow = df_rem.sort_values(by='pct_in', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='pct_in', y='country', data=top_inflow, palette='Blues_d')
plt.title('Top 10 Countries by Remittance Inflow as % of GDP')
plt.xlabel('Inflow (% of GDP)')
plt.ylabel('Country')
plt.tight_layout()
plt.grid(True)
plt.show(block = True)

top_outflow = df_rem.sort_values(by='pct_out', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='pct_out', y='country', data=top_outflow, palette='Oranges_d')
plt.title('Top 10 Countries by Remittance Outflow as % of GDP')
plt.xlabel('Outflow (% of GDP)')
plt.ylabel('Country')
plt.tight_layout()
plt.grid(True)
plt.show(block = True)


##### mirror scatterplot
df_sorted = df_rem.sort_values('inflow', ascending=False).head(25)
x = range(len(df_sorted))

plt.figure(figsize=(14, 6))
plt.barh(x, df_sorted['pct_in'], color='green', label='Inflow')
plt.barh(x, -df_sorted['pct_out'], color='red', label='Outflow')
plt.yticks(x, df_sorted['country'])
plt.xlabel('Percentage of GDP')
plt.title('Inflow vs Outflow of Remittances (% of GDP)')
plt.axvline(0, color='black')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show(block = True)