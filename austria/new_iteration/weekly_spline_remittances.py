import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

## globals
fixed_vars = ['agent_id', 'country', 'sex']

## inflation correction
inflation = pd.read_excel("C:\\Data\\economic\\annual_inflation_clean.xlsx").query("Country == 'Austria' & year >= 2010")
inflation.rename(columns = {'hcpi' : 'rate'}, inplace = True)
inflation['hcpi'] = 100
for year in tqdm(inflation.year.unique()[1:]):
    inflation.loc[inflation.year == year, 'hcpi'] = (inflation.loc[inflation.year == year - 1, 'hcpi'].item() *
                                                     (1 + inflation.loc[inflation.year == year, 'rate'].item() / 100))
inflation['hcpi'] = inflation['hcpi'] / 100

## load simulated population
df = pd.read_pickle("c:\\data\\population\\austria\\simulated_migrants_populations_2010-2024.pkl")

# Reshape the dataframe
result = df.melt(id_vars=["agent_id", "country", "sex"], var_name="year", value_name="age")
result["year"] = result["year"].astype(int)
result = result.groupby(["country", "year", "sex"])["age"].count().reset_index()
result.columns = ["country", "year", "sex", "count"]
result = pd.pivot_table(result, index=["country", "year"], columns='sex', values='count')
result['sex_diff'] = abs(result['male'] - result['female'])/(result['male'] + result['female'])
result.drop(columns = ['male', 'female'], inplace = True)
result.reset_index(inplace = True)

df = df[fixed_vars + [str(x) for x in range(2010, 2026)]]
df.columns = fixed_vars + [str(x) for x in range(2010, 2026)]
df = pd.melt(df, id_vars=fixed_vars, value_vars=df.columns[3:],
             value_name='age', var_name='year')
df['year'] = df['year'].astype(int)
df = df.merge(result, on = ['country', 'year'], how = 'left')
df.fillna(0, inplace = True)

## load quarter remittances info
# df_rem_quarter = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
# df_rem_quarter = df_rem_quarter[(df_rem_quarter.group != 0) & (df_rem_quarter.country.isin(df.country.unique().tolist()))]
# for year in tqdm(df_rem_quarter.year.unique()):
#     df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances'] = (df_rem_quarter.loc[df_rem_quarter.year == year, 'remittances']/
#                                                                       inflation[inflation.year == year]['hcpi'].item())
# df_rem_quarter['exp_population'] = df_rem_quarter['remittances'] / 450
# df_rem_quarter['probability'] = df_rem_quarter['exp_population'] / df_rem_quarter['population']
# df_rem_quarter['probability'] = df_rem_quarter['probability'].clip(0,1)

## load yearly remittances info
df_rem_year = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel.xlsx")
df_rem_year = df_rem_year[(df_rem_year.group != 0) & (df_rem_year.country.isin(df.country.unique().tolist()))]
df_rem_year.rename(columns={"mln_euros" : "remittances", "pop" : "population"}, inplace = True)
df_rem_year['remittances'] *= 1_000_000
for year in tqdm(df_rem_year.year.unique()):
    df_rem_year.loc[df_rem_year.year == year, 'remittances'] = (df_rem_year.loc[df_rem_year.year == year, 'remittances']/
                                                                      inflation[inflation.year == year]['hcpi'].item())
df_rem_year['exp_population'] = df_rem_year['remittances'] / (4 * 450)
df_rem_year['probability'] = df_rem_year['exp_population'] / df_rem_year['population']
df_rem_year['probability'] = df_rem_year['probability'].clip(0,1)

cols = ['date', 'country', 'population', 'remittances']

# Function to get quarter end date
# def get_quarter_end(year, quarter):
#     if quarter == 1:
#         return f"{year}-03-31"
#     elif quarter == 2:
#         return f"{year}-06-30"
#     elif quarter == 3:
#         return f"{year}-09-30"
#     elif quarter == 4:
#         return f"{year}-12-31"
#
# # Create 'date' column
# df_rem_quarter['date'] = df_rem_quarter.apply(lambda row: get_quarter_end(row['year'], row['quarter']), axis=1)
# df_rem_quarter['date'] = pd.to_datetime(df_rem_quarter['date'])
#
# # Extract quarterly data
# quarterly_data = df_rem_quarter.reset_index()[cols]
#
# # Convert dates to numerical values
# quarterly_data['time'] = (quarterly_data['date'] - quarterly_data['date'].min()).dt.days
#
# # Create weekly dates
# start_date = quarterly_data['date'].min()
# end_date = quarterly_data['date'].max()
# weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W')
# weekly_times = (weekly_dates - start_date).days

# Create 'date' column
df_rem_year['date'] = df_rem_year['year'].apply(lambda x: pd.to_datetime(x, format='%Y').replace(month=12, day=31))

# Extract quarterly data
yearly_data = df_rem_year.reset_index()[cols]

# Convert dates to numerical values
yearly_data['time'] = (yearly_data['date'] - yearly_data['date'].min()).dt.days

# Create weekly dates
start_date = yearly_data['date'].min()
end_date = yearly_data['date'].max()
weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W')
weekly_times = (weekly_dates - start_date).days

# Perform cubic spline interpolation
df_res = pd.DataFrame()

for country in tqdm(yearly_data.country.unique()):
    data = [weekly_dates, [country] * len(weekly_dates)]
    for col in cols[2:]:
        cs = CubicSpline(yearly_data[yearly_data.country == country]['time'],
                         yearly_data[yearly_data.country == country][col])
        vals = cs(weekly_times)
        data.append(vals)
    dict_country = dict(zip(cols, data))
    country_df = pd.DataFrame(dict_country)
    df_res = pd.concat([df_res, country_df])
df_res['population'] = df_res['population'].map(np.ceil).astype(int)

df_res.to_csv("c:\\data\\my_datasets\\weekly_remittances_austria.csv", index = False)

def plot_remittances_country(country):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(yearly_data[yearly_data.country == country]['date'],
                yearly_data[yearly_data.country == country]['remittances'],
                color='red', label='Quarterly Data')
    plt.plot(df_res[df_res.country == country]['date'],
             df_res[df_res.country == country]['remittances'],
             label='Cubic Spline Interpolation',
             marker = 'x', markersize=2.5)
    plt.xlabel('Date')
    plt.ylabel('Remittances')
    plt.title(f'Quarterly to Weekly Interpolation, {country}')
    plt.legend()
    plt.grid()
    plt.show(block = True)

plot_remittances_country('Italy')