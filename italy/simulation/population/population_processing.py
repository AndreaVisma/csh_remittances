
 import pandas as pd
 import os
 from tqdm import tqdm
 from utils import dict_names
 import numpy as np

age_groups = ['Less than 5 years', 'From 5 to 9 years', 'From 10 to 14 years',
       'From 15 to 19 years', 'From 20 to 24 years', 'From 25 to 29 years',
       'From 30 to 34 years', 'From 35 to 39 years', 'From 40 to 44 years', 'From 45 to 49 years',
       'From 50 to 54 years', 'From 55 to 59 years','From 60 to 64 years', 'From 65 to 69 years',
       'From 70 to 74 years', 'From 75 to 79 years', 'From 80 to 84 years', 'From 85 to 89 years',
       'From 90 to 94 years', 'From 95 to 99 years', '100 years or over']
 cols = ['citizenship', 'age_group', 'TIME', 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
         2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
         2020, 2021, 2022]

## process stock, emigration and immigration data

def preprocess_flows(flow_type, base_path='c:\\data\\migration\\italy\\'):

    filename = os.path.join(base_path, f"{flow_type}_italy.xlsx")
    dict_sheets = {'Sheet 1': 'male', 'Sheet 2' : 'female'}

    df = pd.DataFrame([])
    for sheet in tqdm(dict_sheets.keys()):
        df_males = pd.read_excel(filename, sheet_name=sheet, skiprows = 8, skipfooter=3).T.iloc[2:]
        df_males.columns = cols
        df_males = df_males[~df_males.citizenship.isna()][['citizenship', 'age_group', 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,2020, 2021, 2022]]
        df_males = df_males.melt(id_vars=["citizenship", "age_group"],var_name="year",value_name="count")
        df_males = df_males[df_males.age_group.isin(age_groups)]
        df_males['sex'] = dict_sheets[sheet]
        df_males.citizenship =df_males.citizenship.apply(lambda x: x.replace("*", ""))
        df_males.citizenship = df_males.citizenship.map(dict_names)
        df_males.dropna(inplace = True)
        df = pd.concat([df, df_males])

    for year in tqdm(df.year.unique()):
        df_year = df[df.year == year]
        df_year.to_csv(base_path + f"{flow_type}_{year}.csv", index = False)

for flow in tqdm(['immigration', 'emigration']):
    preprocess_flows(flow)


 def preprocess_stock(year):
     df_mig_pop = pd.read_excel("C:\\Data\\migration\\bilateral_migration_undesa.xlsx",
                                sheet_name='Table 1', skiprows=10).iloc[:, 1:]
     df_mig_pop.columns = ['dest', 'coverage', 'dtype', 'code', 'origin', 'codeor',
                           1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024,
                           '1990_m', '1995_m', '2000_m', '2005_m', '2010_m', '2015_m', '2020_m', '2024_m',
                           '1990_f', '1995_f', '2000_f', '2005_f', '2010_f', '2015_f', '2020_f', '2024_f']
     df_ita = df_mig_pop[df_mig_pop.dest == "Italy"][['origin', f'{year}_m', f'{year}_f']]
     df_ita.origin = df_ita.origin.apply(lambda x: x.replace("*", ""))
     df_ita.origin = df_ita.origin.map(dict_names)
     df_ita.dropna(inplace=True)
     df_ita.columns = ['citizenship', 'male', 'female']
     df_ita = pd.melt(df_ita, id_vars=["citizenship"], var_name="sex", value_name="count")
     df_ita = df_ita[['citizenship', 'count']].groupby('citizenship').sum().reset_index()

     ## divide stock in age groups based on percentage arrivals before and including year
     years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
     years = [x for x in years if x <= 2010]
     df_pct = pd.DataFrame([])
     for year in years:
         df = pd.read_csv(os.path.join("c:\\data\\migration\\italy\\", f"immigration_{year}.csv"))
         df['count'] = df['count'].replace(':', 0).astype(int)
         for country in df.citizenship.unique():
             df.loc[df.citizenship == country, 'pct'] = 100 * (df.loc[df.citizenship == country, 'count'] /
                                                         df.loc[df.citizenship == country, 'count'].sum())
         df_pct = pd.concat([df_pct, df])
     df_pct = df_pct.groupby(['age_group', 'sex', 'citizenship']).mean().reset_index().drop(columns = ['year', 'count'])

     df_ita = df_ita.merge(df_pct, on = ['citizenship'])
     df_ita['count'] = df_ita['count'] * df_ita['pct'] * 0.01
     df_ita['count'] = df_ita['count'].map(np.round).astype('int')
     df_ita.drop(columns = 'pct', inplace = True)

     df_ita.to_csv(f'c:\\data\\migration\\italy\\baseline_stock_{year}.csv', index=False)

 preprocess_stock(2010)

 def load_flow_data(year, flow_type, countries, base_path='c:\\data\\migration\\italy\\'):
     """
     Load flow data for a given year.

     Parameters:
       year (int): The year of the data.
       flow_type (str): Either 'immigration' or 'emigration'.
       base_path (str): Directory where CSV files are stored.

     Returns:
       pd.DataFrame: A DataFrame indexed by [age_group, sex, citizenship].
     """
     filename = os.path.join(base_path, f"{flow_type}_{year}.csv")
     # Expecting CSV files with columns: age_group, sex, citizenship, count
     df = pd.read_csv(filename)
     df = df[df.citizenship.isin(countries)]
     # Set a MultiIndex based on the grouping variables
     df = df.set_index(['age_group', 'sex', 'citizenship'])
     # It is convenient to have just one column with the count:
     df = df[['count']]
     df['count'] = df['count'].replace(':', 0).astype(int)
     return df

 # def update_stock(previous_stock, immigration, emigration):
 #     """
 #     Update the migrant stock using flows.
 #
 #     Parameters:
 #       previous_stock (pd.DataFrame): Stock in the previous year (indexed by age_group, sex, citizenship).
 #       immigration (pd.DataFrame): Immigration flows for the current year.
 #       emigration (pd.DataFrame): Emigration flows for the current year.
 #
 #     Returns:
 #       pd.DataFrame: The updated stock.
 #
 #     Note: This function assumes that the indices in the dataframes line up.
 #     """
 #     new_stock = previous_stock.add(immigration, fill_value=0) \
 #         .subtract(emigration, fill_value=0)
 #
 #     # Ensure counts are not negative (optional, depending on your data)
 #     new_stock['count'] = new_stock['count'].clip(lower=0)
 #
 #     return new_stock

 def update_stock(previous_stock, immigration, emigration):
     """
     Update the migrant stock using flows, mortality, and natality.

     Parameters:
       previous_stock (pd.DataFrame): Stock in the previous year (indexed by age_group, sex, citizenship).
       immigration (pd.DataFrame): Immigration flows for the current year.
       emigration (pd.DataFrame): Emigration flows for the current year.

     Returns:
       pd.DataFrame: The updated stock.
     """
     # Apply mortality (10 per 1000)
     survivors = previous_stock.copy()
     deaths = (survivors['count'] * 0.01).round().astype(int)
     survivors['count'] = (survivors['count'] - deaths).clip(lower=0)

     # Calculate births (7 per 1000 of survivors)
     total_survivors = survivors['count'].sum()
     births_total = int(round(total_survivors * 0.007))  # 7 per 1000

     if total_survivors > 0 and births_total > 0:
         # Distribute births by citizenship
         all_citizenships = survivors.index.get_level_values('citizenship').unique()
         citizenship_totals = survivors.groupby('citizenship')['count'].sum().reindex(all_citizenships, fill_value=0)
         birth_shares = citizenship_totals / total_survivors
         births_per_citizenship = (birth_shares * births_total).round().astype(int)

         # Adjust for rounding discrepancies
         total_assigned = births_per_citizenship.sum()
         if total_assigned != births_total:
             diff = births_total - total_assigned
             max_citizenship = citizenship_totals.idxmax()
             births_per_citizenship[max_citizenship] += diff

         # Create births entries
         births_data = []
         for citizenship in births_per_citizenship.index:
             total_births = births_per_citizenship[citizenship]
             if total_births <= 0:
                 continue
             male_births = total_births // 2
             female_births = total_births - male_births
             births_data.append(('Less than 5 years', 'male', citizenship, male_births))
             births_data.append(('Less than 5 years', 'female', citizenship, female_births))

         # Add births to survivors
         if births_data:
             births_df = pd.DataFrame(
                 births_data,
                 columns=['age_group', 'sex', 'citizenship', 'count']
             ).set_index(['age_group', 'sex', 'citizenship'])
             survivors = survivors.add(births_df, fill_value=0)

     # Apply immigration and emigration
     new_stock = survivors.add(immigration, fill_value=0).subtract(emigration, fill_value=0)
     new_stock['count'] = new_stock['count'].clip(lower=0).astype(int)

     return new_stock

 def compute_stocks(baseline_year=2010, start_year=2008, end_year=2022, data_path='c:\\data\\migration\\italy\\'):
     """
     Compute the migrant stock for each year from start_year to end_year,
     given a baseline stock (divided by sex and citizenship) and annual flows.

     Parameters:
       baseline_year (int): The year with known stock.
       start_year (int): Earliest year for estimation.
       end_year (int): Latest year for estimation.
       data_path (str): Directory where CSV files are stored.

     Returns:
       dict: Dictionary with year as key and the corresponding stock DataFrame as value.
     """
     # Load baseline stock for baseline_year.
     baseline_file = os.path.join(data_path, f"baseline_stock_{baseline_year}.csv")
     stock = pd.read_csv(baseline_file)
     countries = stock.citizenship.unique().tolist()
     stock = stock.set_index(['age_group', 'sex', 'citizenship'])
     stock = stock[['count']]

     stocks = {baseline_year: stock}

     # Compute stocks for years after the baseline.
     for year in tqdm(range(baseline_year + 1, end_year + 1)):
         # Load flows for the year; if file not found, assume zero flows.
         try:
             imm_flow = load_flow_data(year, 'immigration', countries = countries, base_path=data_path)
         except FileNotFoundError:
             imm_flow = pd.DataFrame(columns=['count'])
         try:
             emi_flow = load_flow_data(year, 'emigration', countries = countries, base_path=data_path)
         except FileNotFoundError:
             emi_flow = pd.DataFrame(columns=['count'])

         previous_stock = stocks[year - 1]
         new_stock = update_stock(previous_stock, imm_flow, emi_flow)
         stocks[year] = new_stock

     # Compute stocks for years before the baseline, if needed.
     for year in tqdm(range(baseline_year - 1, start_year - 1, -1)):
         # For reverse update, assume:
         # Stock[t] = Stock[t+1] - Immigration[t+1] + Emigration[t+1]
         try:
             imm_flow_next = load_flow_data(year + 1, 'immigration', countries = countries, base_path=data_path)
         except FileNotFoundError:
             imm_flow_next = pd.DataFrame(columns=['count'])
         try:
             emi_flow_next = load_flow_data(year + 1, 'emigration', countries = countries, base_path=data_path)
         except FileNotFoundError:
             emi_flow_next = pd.DataFrame(columns=['count'])

         next_stock = stocks[year + 1]
         previous_stock = next_stock.subtract(imm_flow_next, fill_value=0) \
             .add(emi_flow_next, fill_value=0)
         stocks[year] = previous_stock

     return stocks


 # Adjust these parameters as needed
 baseline_year = 2010
 start_year = 2008
 end_year = 2022
 data_path = 'c:\\data\\migration\\italy\\'  # your data directory

 stocks_by_year = compute_stocks(baseline_year=baseline_year, start_year=start_year, end_year=end_year,
                                 data_path=data_path)

 # For example, to display the migrant stock for 2020:
 print("Migrant stock in 2020:")
 print(stocks_by_year[2020].sort_index())

 df_all = pd.DataFrame([])
 for year, df in tqdm(stocks_by_year.items(), total = len(stocks_by_year.keys())):
     df['year'] = year
     df_all = pd.concat([df_all, df])
 df_all.to_csv('c:\\data\\migration\\italy\\estimated_stocks_new.csv')
