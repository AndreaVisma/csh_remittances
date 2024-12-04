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
df_rem_quarter['remittances_per_person'] = df_rem_quarter['remittances'] / df_rem_quarter['population']
#####
# yearly values
df_year = df_rem_quarter.groupby(['country', 'year'])[['remittances', 'remittances_per_person']].sum().reset_index()
mean_df = df_year.groupby('country')[['remittances', 'remittances_per_person']].mean().reset_index()
above_1m = mean_df[mean_df.remittances > 100_000]

# Create the histogram using seaborn
plt.figure(figsize=(10, 6))
sns.histplot(above_1m['remittances_per_person'], kde=True, bins=50)

# Add labels and title
plt.xlabel('Remittances per Person')
plt.ylabel('Frequency')
plt.title('Histogram of Remittances per Person')
plt.grid()
# Save and show the plot
plt.savefig(outfolder + 'histogram_remittances_per_person.pdf', dpi=300)
plt.show(block = True)