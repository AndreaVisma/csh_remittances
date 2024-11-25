import pandas as pd
import matplotlib.pyplot as plt

#population
df_pop = pd.read_excel("c:\\data\\migration\\austria\\quarterly_population_clean.xlsx")

#remittances
df_rem = pd.read_excel("c:\\data\\remittances\\austria\\quarterly_remittances_sent_clean.xlsx")
df = df_pop.merge(df_rem, on = ['country', 'year', 'quarter'], how = 'inner')

#natural disasters
df_nd = pd.read_excel("C:\\Data\\natural_disasters\\emdat_country_type_quarterly.xlsx")
#clean dates
df_nd[['Start Year','Start Month', 'Start Day']] = (
    df_nd[['Start Year','Start Month', 'Start Day']].fillna(1).astype(int))
df_nd.rename(columns = dict(zip(['Start Year','Start Month', 'Start Day'],
                              ["year", "month", "day"])), inplace = True)
df_nd["date_start"] = pd.to_datetime(df_nd[["year", "month", "day"]])
df_nd.drop(columns = ["year", "month", "day", "quarter"], inplace = True)
df_nd['year'] = df_nd['date_start'].dt.year
df_nd['quarter'] = df_nd['date_start'].dt.quarter
df_nd['quarter_after_1_month'] = df_nd['date_start'].apply(lambda x: 1 + (x.month)//3)
df_nd.rename(columns = {'Country' : 'country'}, inplace = True)

## country population
df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")
#merge
df_nd = df_nd.merge(df_pop_country, on=['country', 'year'], how = 'left')

#percentage affected dataframe
df_nd_pct = df_nd.copy()
cols = ['Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected']
for col in cols:
    df_nd_pct[col] = 100 * df_nd_pct[col] / df_nd_pct['population']
df_nd_pct.dropna(inplace = True)
df_nd_pct.sort_values('total affected', ascending = False, inplace = True)
df_nd_pct['total affected'].hist()
plt.show(block = True)

#export a dataset to run regression analyses in R
cols = ['country', 'Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected', 'year',
       'quarter_after_1_month']
df_group = df_nd_pct[cols].groupby(['country', 'year', 'quarter_after_1_month']).sum().reset_index()
df_all = df.merge(df_group, left_on = ['country', 'year', 'quarter'],
                  right_on = ['country', 'year', 'quarter_after_1_month'], how = 'left')
df_all.fillna(0, inplace = True)
df_all.to_excel("c:\\data\\my_datasets\\panel_rem_disaster_quarterly_shifted.xlsx", index = False)

cols = ['country', 'Animal incident', 'Drought', 'Earthquake', 'Epidemic',
       'Extreme temperature', 'Flood', 'Glacial lake outburst flood', 'Impact',
       'Infestation', 'Mass movement (dry)', 'Mass movement (wet)', 'Storm',
       'Volcanic activity', 'Wildfire', 'total affected', 'year',
       'quarter']
df_group = df_nd_pct[cols].groupby(['country', 'year', 'quarter']).sum().reset_index()
df_all = df.merge(df_group, on = ['country', 'year', 'quarter'], how = 'left')
df_all.fillna(0, inplace = True)
df_all.to_excel("c:\\data\\my_datasets\\panel_rem_disaster_quarterly.xlsx", index = False)