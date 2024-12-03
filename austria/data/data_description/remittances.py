import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import country_converter as coco
cc = coco.CountryConverter()

def millions_formatter(x, pos):
    return f'{x/1e6:.0f} mln'

outfolder = ".\\austria\\plots\\plots_for_paper\\remittances\\"

## inflation correction
inflation = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx").query("Country == 'Austria' & year >= 2010")
inflation.rename(columns = {'hcpi' : 'rate'}, inplace = True)
inflation['hcpi'] = 100
for year in tqdm(inflation.year.unique()[1:]):
    inflation.loc[inflation.year == year, 'hcpi'] = (inflation.loc[inflation.year == year - 1, 'hcpi'].item() *
                                                     (1 + inflation.loc[inflation.year == year, 'rate'].item() / 100))
inflation['hcpi'] = inflation['hcpi'] / 100

## load remittances info
df_rem_quarter = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
for year in tqdm(df_rem_quarter.year.unique()):
    df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances'] = (df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())
df_rem_quarter['exp_population'] = df_rem_quarter['remittances'] / 450
df_rem_quarter['probability'] = df_rem_quarter['exp_population'] / df_rem_quarter['population']

#####
# yearly values
df_year = df_rem_quarter.groupby(['country', 'year'])['remittances'].sum().reset_index()
agg_df = df_year.groupby('country')['remittances'].sum().reset_index()

##
mean_df = df_year.groupby('country')['remittances'].mean().reset_index()
above_50m = mean_df[mean_df.remittances > 50_000_000]

# Mean and median
mean_country_remit = mean_df['remittances'].mean()
median_country_remit = mean_df['remittances'].median()
mean_year_remit = df_year.groupby('year')['remittances'].sum().mean()
median_year_remit = df_year.groupby('year')['remittances'].sum().median()

# Skewness and kurtosis
skewness = skew(mean_df['remittances'])
kurt = kurtosis(mean_df['remittances'])

print(f"Mean _country Remittances: {mean_country_remit}")
print(f"Median _country Remittances: {median_country_remit}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurt}")

print(f"Mean _year Remittances: {mean_year_remit}")
print(f"Median _year Remittances: {median_year_remit}")

def gini_func(series):
    """Calculate the Gini coefficient of a numpy array."""
    array = series.to_numpy()
    array = array.flatten()
    array = array + 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

# Calculate Gini coefficient
gini = gini_func(df_year[df_year.year == 2020]['remittances'])
print(f"Gini Coefficient: {gini}")




sns.kdeplot(agg_df)
plt.show(block = True)

## line chart
total_remit = df_year.groupby('year')['remittances'].sum().reset_index()

fig = plt.figure(figsize=(10,6))
plt.plot(total_remit['year'], total_remit['remittances'], marker='o')
plt.title('Total Remittances Sent from Austria Over Time')
plt.xlabel('Year')
plt.ylabel('Total Remittances (euros)')
plt.xticks(total_remit['year'])
plt.grid(True)
formatter = FuncFormatter(millions_formatter)
plt.gca().yaxis.set_major_formatter(formatter)
fig.savefig(outfolder + 'total_remittances.jpg', dpi=fig.dpi, bbox_inches = 'tight')
plt.show(block = True)

## stacked bar chart
# Determine top countries each year
top_countries = df_year[df_year['year'] == 2023].sort_values('remittances', ascending=False).head(9)['country'].tolist()

df_year['category'] = df_year.apply(lambda x: x['country'] if x['country'] in top_countries else 'Other', axis=1)
grouped = df_year.groupby(['year', 'category'])['remittances'].sum().reset_index()
pivot = grouped.pivot(index='year', columns='category', values='remittances').fillna(0)

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(12,8))
pivot.plot(kind='bar', stacked=True, ax = ax)
plt.title('Remittances to Top Countries and Others Over Time')
plt.xlabel('Year')
plt.ylabel('Remittances (euros)')
plt.legend(title='Country', bbox_to_anchor=(1,1))
plt.grid()
formatter = FuncFormatter(millions_formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.xticks(rotation=360)
fig.savefig(outfolder + 'remittances_stacked.jpg', dpi=fig.dpi, bbox_inches = 'tight')
plt.show(block = True)

#heatmap
top_countries = df_year[df_year['year'] == 2023].sort_values('remittances', ascending=False).head(10)['country'].tolist()
heatmap_data = df_year[df_year.country.isin(top_countries)].pivot(index='country', columns='year', values='remittances').fillna(0)

# Plot heatmap
plt.figure(figsize=(12,12))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5)
plt.title('Remittances Distribution Across Countries and Years')
plt.xlabel('Year')
plt.ylabel('Country')
plt.yticks(rotation=0)
fig.savefig(outfolder + 'remittances_heatmap.jpg', dpi=fig.dpi, bbox_inches = 'tight')
plt.show(block = True)

#lorenz curve
yearly_data = df_year.groupby('country')['remittances'].mean().reset_index()
remittances = yearly_data[yearly_data.remittances > 1000]['remittances'].sort_values()
cum_rem = remittances.cumsum()/remittances.sum()
fig, ax = plt.subplots(figsize=(12,8))
cum_rem.plot(kind = 'bar', ax = ax, label = 'Remittances')
ax.plot([0, len(remittances) - 1], [0, 1], color='k', linestyle='--', label='Perfect Equality')
plt.title(f'Lorenz Curve for Remittances Distribution')
plt.xlabel('Countries')
plt.ylabel('Share of total remittances')
plt.legend()
plt.grid()
ax.get_xaxis().set_ticks([])
fig.savefig(outfolder + 'remittances_lorenz_curve.pdf', dpi=fig.dpi, bbox_inches = 'tight')
plt.show(block=True)


# map
iso3_codes = cc.pandas_convert(series=df_year.country, to='ISO3')
df_year['iso_code'] = iso3_codes

# Create choropleth map
fig = px.choropleth(df_year[df_year.iso_code != 'not found'],
                    locations='iso_code',
                    locationmode='ISO-3',
                    color='remittances',
                    hover_name='country',
                    animation_frame='year',
                    title='Remittances Sent from Austria by Country',
                    color_continuous_scale='Viridis')
fig.write_html(outfolder + 'remittances_map.html')
fig.show()