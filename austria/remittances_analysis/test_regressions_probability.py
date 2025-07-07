import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

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
df_rem_quarter = df_rem_quarter[(df_rem_quarter.group != 0) & (df_rem_quarter.country.isin(df.country.unique().tolist()))]
for year in tqdm(df_rem_quarter.year.unique()):
    df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances'] = (df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())
df_rem_quarter['exp_population'] = df_rem_quarter['remittances'] / 450
df_rem_quarter['probability'] = df_rem_quarter['exp_population'] / df_rem_quarter['population']

## fit regression
cols = ['quarter', 'total affected']
X = df_rem_quarter[cols] # Features
y = df_rem_quarter.probability # Target variable

X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
results.summary()