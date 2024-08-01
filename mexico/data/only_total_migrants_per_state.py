import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import geopandas
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

folder = "c:\\data\\migration\\mexico\\data_ime\\"

#############
## DO THIS ONLY ONCE!!
###############

# for year in tqdm([2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]):
#     with zipfile.ZipFile(folder + f"Matriculas Rep. Mex.{year}.zip", 'r') as zip_ref:
#         zip_ref.extractall(folder + f"\\{year}_ext")

##########################
#######################

df = pd.DataFrame([])

#2013
year = 2013
folder_year = folder + f"{year}_ext\\Matriculas Rep. Mex.{year}"
print(f"Processing {year} ...")
for state in tqdm(os.listdir(folder_year)):
    folder_state = folder_year + f"\\{state}"
    file = [x for x in os.listdir(folder_state) if "porEdoUsa" in x and "._" not in x][0]
    file_state = folder_state + f"\\{file}"

    df_state = pd.read_excel(file_state, skiprows=8, skipfooter=7, usecols="B:D")
    df_state.rename(columns={'Estado de Origen' : 'us_state',
                             df_state.columns[1] : 'nr_registered',
                             df_state.columns[2] : 'pct_registered'}, inplace = True)
    df_state['mex_state'] = state
    df_state['year'] = year
    df = pd.concat([df, df_state])
#2014
year = 2014
print(f"Processing {year} ...")
folder_year = folder + f"{year}_ext\\Matriculas Rep. Mex.{year}"
for state in tqdm(os.listdir(folder_year)):
    folder_state = folder_year + f"\\{state}"
    file = [x for x in os.listdir(folder_state) if "edousa" in x and "._" not in x][0]
    file_state = folder_state + f"\\{file}"

    df_state = pd.read_excel(file_state, skiprows=10, skipfooter=7, usecols="B:D")
    df_state.rename(columns={df_state.columns[0] : 'us_state',
                             df_state.columns[1] : 'nr_registered',
                             df_state.columns[2] : 'pct_registered'}, inplace = True)
    df_state['mex_state'] = state
    df_state['year'] = year
    df = pd.concat([df, df_state])
#2015
years = [2015, 2016, 2017]
for year in years:
    print(f"Processing {year} ...")
    folder_year = folder + f"{year}_ext\\Matriculas Rep. Mex.{year}"
    for state in tqdm(os.listdir(folder_year)):
        folder_state = folder_year + f"\\{state}"
        file = [x for x in os.listdir(folder_state) if "Edo_USA" in x and "._" not in x][0]
        file_state = folder_state + f"\\{file}"

        df_state = pd.read_excel(file_state, skiprows=10, skipfooter=7, usecols="B:D")
        df_state.rename(columns={df_state.columns[0] : 'us_state',
                                 df_state.columns[1] : 'nr_registered',
                                 df_state.columns[2] : 'pct_registered'}, inplace = True)
        df_state['mex_state'] = state
        df_state['year'] = year
        df = pd.concat([df, df_state])
#2018, 2020, 2021, 2022
years = [2018, 2020, 2021, 2022]
for year in years:
    print(f"Processing {year} ...")
    folder_year = folder + f"{year}_ext\\Matriculas Rep. Mex.{year}"
    for state_file in tqdm(os.listdir(folder_year)):
        if year == 2018:
            state = state_file.replace(".xlsx", "")
        if year > 2019:
            state = state_file.replace(f" {year}.xlsx", "")
        file_state = folder_year + f"\\{state_file}"

        xl = pd.ExcelFile(file_state)
        sheet_name = [x for x in xl.sheet_names  if "EdoUsa" in x][0]# see all sheet names
        df_state = pd.read_excel(file_state, sheet_name = sheet_name,
                                 skiprows=10, skipfooter=7, usecols="B:D")
        df_state.rename(columns={df_state.columns[0] : 'us_state',
                                 df_state.columns[1] : 'nr_registered',
                                 df_state.columns[2] : 'pct_registered'}, inplace = True)
        df_state['mex_state'] = state
        df_state['year'] = year
        df = pd.concat([df, df_state])
#2019
year = 2019
print(f"Processing {year} ...")
folder_year = folder + f"{year}_ext\\Matriculas Rep. Mex.{year}"
for state_file in tqdm(os.listdir(folder_year)):
    state = state_file.replace(f" {year}.xlsx", "")
    file_state = folder_year + f"\\{state_file}"

    xl = pd.ExcelFile(file_state)
    sheet_name = [x for x in xl.sheet_names if "Edomexgral" in x][0]  # see all sheet names
    df_state = pd.read_excel(file_state, sheet_name=sheet_name,
                             skiprows=10, skipfooter=7, usecols="B:D")
    df_state.rename(columns={df_state.columns[0]: 'mex_state',
                             df_state.columns[1]: 'nr_registered',
                             df_state.columns[2]: 'pct_registered'}, inplace=True)
    df_state['us_state'] = state
    df_state['year'] = year
    df_state = df_state[df.columns.tolist()]
    df = pd.concat([df, df_state])

##clean up a bit
df.nr_registered.fillna(0, inplace = True)
df.dropna(inplace = True)
df = df[df.us_state != "Total"]
df = df[df.mex_state != "Total"]
mex_states = ['aguascalientes', 'baja california', 'baja california sur',
       'campeche', 'chiapas', 'chihuahua', 'ciudad de mxico', 'coahuila',
       'colima', 'durango', 'estado de mxico', 'guanajuato', 'guerrero',
       'hidalgo', 'jalisco', 'michoacn', 'morelos', 'nayarit',
       'nuevo len', 'puebla', 'quertaro', 'quintana roo',
       'san luis potos', 'sinaloa', 'sonora', 'tabasco', 'tamaulipas',
       'tlaxcala', 'veracruz', 'yucatn', 'zacatecas', 'san_luis_potosi',
       'oaxaca', 'Aguascalientes', 'Baja California Sur',
       'Baja California', 'Campeche', 'Chiapas', 'Chihuahua',
       'Ciudad de México', 'Coahuila', 'Colima', 'Durango',
       'Estado de México', 'Guanajuato', 'Guerrero', 'Hildalgo',
       'Jalisco', 'Michoacán', 'Morelos', 'Nayarit', 'Nuevo León',
       'Oaxaca', 'Puebla', 'Querétaro', 'Quintana Roo', 'San Luis Potosí',
       'Sinaloa', 'Sonora', 'Tabasco', 'Tamaulipas', 'Tlaxcala',
       'Veracruz', 'Yucatán', 'Zacatecas', 'Hidalgo']
mex_states_nice = ['Aguascalientes', 'Baja California', 'Baja California Sur',
       'Campeche', 'Chiapas', 'Chihuahua', 'Ciudad de México', 'Coahuila',
       'Colima','Durango', 'Estado de México', 'Guanajuato',
       'Guerrero', 'Hidalgo', 'Jalisco', 'Michoacán', 'Morelos',
       'Nayarit', 'Nuevo León', 'Puebla', 'Querétaro',
       'Quintana Roo', 'San Luis Potosí', 'Sinaloa', 'Sonora', 'Tabasco',
       'Tamaulipas', 'Tlaxcala', 'Veracruz', 'Yucatán', 'Zacatecas',
       'San Luis Potosí', 'Oaxaca', 'Aguascalientes', 'Baja California Sur', 'Baja California',
       'Campeche', 'Chiapas', 'Chihuahua',
       'Ciudad de México', 'Coahuila', 'Colima', 'Durango', 'Estado de México', 'Guanajuato',
       'Guerrero', 'Hidalgo', 'Jalisco', 'Michoacán', 'Morelos',
       'Nayarit', 'Nuevo León', 'Oaxaca', 'Puebla', 'Querétaro',
       'Quintana Roo', 'San Luis Potosí', 'Sinaloa', 'Sonora', 'Tabasco',
       'Tamaulipas', 'Tlaxcala', 'Veracruz', 'Yucatán', 'Zacatecas', 'Hidalgo']
dict_mex_names  = dict(zip(mex_states, mex_states_nice))
df.mex_state = df.mex_state.map(dict_mex_names)

df.to_excel("c:\\data\\migration\\mexico\\migrants_mex_us_matriculas.xlsx", index = False)
#check
df_group = df[['nr_registered', 'mex_state', 'year']].groupby(['mex_state', 'year'], as_index = False).sum()
df_group.to_excel("c:\\data\\migration\\mexico\\migrants_mex_state_aggregate.xlsx", index = False)

fig = px.line(df_group[~df_group.year.isin([2014, 2019])], x = 'year', y = 'nr_registered', color='mex_state')
fig.update_layout(title = "Nr Mexican migrants registered at a consulate over time. by state of origin")
fig.write_html(os.getcwd() + "\\mexico\\data\\plots\\matriculas_per_state_overtime.html")
fig.show()

## by us state
df_group = df[['nr_registered', 'us_state', 'year']].groupby(['us_state', 'year'], as_index = False).sum()
df_group.to_excel("c:\\data\\migration\\mexico\\migrants_us_state_aggregate.xlsx", index = False)

fig = px.line(df_group[~df_group.year.isin([2014, 2019])], x = 'year', y = 'nr_registered', color='us_state')
fig.update_layout(title = "Nr Mexican migrants registered at a consulate over time. by state of destination")
fig.write_html(os.getcwd() + "\\mexico\\data\\plots\\matriculas_per_US_state_overtime.html")
fig.show()