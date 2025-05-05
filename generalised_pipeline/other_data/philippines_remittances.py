
import pandas as pd
from tqdm import tqdm
from utils import dict_names

file = "C:\\Data\\remittances\\philippines_remittances.xls"
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

xl = pd.ExcelFile(file)
sheet_names = xl.sheet_names[2:-4]  # see all sheet names

df_list = []
for sheet in tqdm(sheet_names):
    year = sheet[-4:]
    df = pd.read_excel(file, sheet_name=sheet, skiprows=9, skipfooter=10, usecols = "F:S")
    df = df[~df.Country.isna()]
    df['destination'] = df['Country'].apply(lambda x: x.strip())
    df = df[df.destination.isin(dict_names.keys())]
    df['destination'] = df['destination'].map(dict_names)
    df = df.iloc[:, 2:]
    df = pd.melt(df, id_vars='destination', value_vars=df.columns[:-1], var_name='month', value_name='remittances')
    df['month_num'] = df['month'].map(month_map)
    df['year'] = int(year)
    df['date'] = pd.to_datetime({
        'year': df['year'],
        'month': df['month_num'],
        'day': 1
    }) + pd.offsets.MonthEnd(0)
    df['origin'] = "Philippines"
    df_list.append(df[['origin', 'destination', 'date', 'remittances']])

df_all = pd.concat(df_list)
df_all['remittances'] *= 1_000

df_all.to_pickle("C:\\Data\\remittances\\Philippines\\phil_remittances_detail.pkl")