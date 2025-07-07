import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

df_conf = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_conflicts.csv")
df_conf['start_week'] = pd.to_datetime(df_conf['start_week'])
df_dis = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_disasters.csv")
df_dis.rename(columns = {"week_start" : "start_week"}, inplace = True)
df_dis['start_week'] = pd.to_datetime(df_dis['start_week'])
df_rem = pd.read_csv("C:\\Data\\my_datasets\\weekly_remittances\\weekly_remittances_austria.csv")
df_rem['date'] = pd.to_datetime(df_rem['date']) + datetime.timedelta(days=1)
df_rem.rename(columns = {"date" : "start_week"}, inplace = True)

df = df_conf.merge(df_dis, on = ['country', 'start_week'], how = 'outer')
df = df.merge(df_rem, on = ['country', 'start_week'], how = 'outer')

df = df[~df.remittances.isna()]
df.fillna(0, inplace = True)
df['year'] = df.start_week.dt.year

df_pop_country = pd.read_excel("c:\\data\\population\\population_by_country_wb_clean.xlsx")

df = df.merge(df_pop_country, on = ['country', 'year'], how = 'left')
df['deaths_pct_pop'] = 100 * df['deaths'] / df["population_y"]
df['total_affected_pct_pop'] = 100 * df['total_affected'] / df["population_y"]
df.drop(columns = "population_y", inplace = True)
df.rename(columns = {"population_x" : "population"}, inplace = True)

df.dropna(inplace = True)
df.to_csv("c:\\data\\my_datasets\\weekly_remittances\\rem_conf_dis_full.csv", index = False)
df = pd.read_csv("c:\\data\\my_datasets\\weekly_remittances\\rem_conf_dis_full.csv")

latam = df[df['country'].isin(["Mexico", "Brazil", "Syria"])]

sns.lineplot(latam, x = "start_week", y = "deaths_pct_pop", hue = "country")
plt.grid()
plt.show(block = True)