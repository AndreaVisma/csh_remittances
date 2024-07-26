import pandas as pd
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

folder = "c:\\data\\remittances\\mexico\\consulados\\"

for year in tqdm([2017, 2018, 2019, 2020, 2021, 2022]):
    with zipfile.ZipFile(folder + f"Consulados_{year}.zip", 'r') as zip_ref:
        zip_ref.extractall(folder + f"\\consulados_{year}_ext")


df = pd.read_csv("c:\\data\\migration\\bilat_mig_sex.csv")
df = df[(df.orig == "MEX") & (df.dest == "USA")]
df = df.groupby('year0', as_index = False).sum()
plt.plot(x = df.year0, y=df.mig_rate)
plt.show(block = True)

#2022 data
df_tot = pd.DataFrame([])
for year in [2018, 2019, 2020, 2021, 2022]:
    folder_year = folder + f"consulados_{year}_ext\\Consulados_{year}\\"

    if year > 2018:
        pbar = tqdm(os.listdir(folder_year))
        for file_name in pbar:
            file = folder_year + file_name
            consulate = file_name.replace(f" {year}.xlsx", "")
            pbar.set_description(f"processing {consulate} registrations ...")
            df = pd.read_excel(file, sheet_name=consulate+"_edomexmun",
                               skiprows = 8, skipfooter=7, usecols="B:D")
            df["registration_consulate"] = consulate
            df["year"] = year
            df = df[df["Municipio de Origen"] != "Total"]
            df.ffill(inplace = True)
            df_tot = pd.concat([df_tot, df])
    if year == 2018:
        pbar = tqdm(os.listdir(folder_year))
        for file_name in pbar:
            file = folder_year + file_name
            consulate = file_name.replace(f"_info_{year}.xlsx", "")
            pbar.set_description(f"processing {consulate} registrations ...")
            df = pd.read_excel(file, sheet_name=consulate + f"_edomexmun{str(year)[-2:]}",
                               skiprows=8, skipfooter=7, usecols="B:D")
            df["registration_consulate"] = consulate
            df["year"] = year
            df = df[df["Municipio de Origen"] != "Total"]
            df.ffill(inplace=True)
    # if year == 2017:
    #     pbar = tqdm(os.listdir(folder_year))
    #     for file_name in pbar:
    #         file = folder_year + file_name
    #         consulate = file_name.replace(f"_info_{year}.xlsx", "")
    #         pbar.set_description(f"processing {consulate} registrations ...")
    #         df = pd.read_excel(file, sheet_name=consulate + f"_edomexmun{str(year)[-2:]}",
    #                            skiprows=8, skipfooter=7, usecols="B:D")
    #         df["registration_consulate"] = consulate
    #         df["year"] = year
    #         df = df[df["Municipio de Origen"] != "Total"]
    #         df.ffill(inplace=True)
            df_tot = pd.concat([df_tot, df])

df_tot.replace('Mcallen', "Mc Allen", inplace = True)
df_tot.to_excel(os.getcwd() + "\\mexico\\data\\consulates_network_2018_2022.xlsx", index = False)

###
gdf = geopandas.read_file("C:\\data\\geo\\georef-mexico-municipality-millesime@public\\georef-mexico-municipality-millesime.shp")
gdf['mun_name'] = gdf['mun_name'].apply(lambda x: ast.literal_eval(x)[0])
gdf.sort_values('mun_name', inplace = True)
gdf = gdf[~gdf.duplicated(['mun_name'])]

# Load the uploaded Excel file to examine its contents
file_path = os.getcwd() + "\\mexico\\data\\consulates_network_2018_2022.xlsx"
df = pd.read_excel(file_path)
df.rename(columns = {'Municipio de Origen': 'mun_name'}, inplace = True)

miss = ['Batopilas','Heroica Villa Tezoatlán de Segura y Luna, Cuna de la Independencia de Oaxaca',
        'Magdalena Jicotlán', 'MonteMorelos', 'San Juan Mixtepec - Distr. 08 -', 'San Juan Mixtepec - Distr. 26 -',
        'San Pedro Mixtepec - Distr. 22 -','San Pedro Mixtepec - Distr. 26 -', 'San Pedro Totolapa',
        'Silao', 'Temósachi', 'Ticumuy', 'Tixpéhual ', 'Villa de Tututepec de Melchor Ocampo',
        'Zacatepec de Hidalgo', 'Zapotitlán del Río']
fix = ['Batopilas de Manuel Gómez Morín', 'Heroica Villa Tezoatlán de Segura y Luna', 'Santa Magdalena Jicotlán',
       'Montemorelos', 'San Juan Mixtepec','San Juan Mixtepec', 'San Juan Mixtepec', 'San Juan Mixtepec',
       'San Pedro Totolápam', 'Silao de la Victoria', 'Temósachic', 'Timucuy',
       'Tixpéhual', 'Melchor Ocampo', 'Zacatepec', 'San Antonio Huitepec']
dict_names = dict(zip(miss, fix))

df.loc[df.mun_name.isin(miss), 'mun_name'] = df.loc[df.mun_name.isin(miss), 'mun_name'].map(dict_names)

#merge
df = df.merge(gdf[['mun_name', 'geometry']], on='mun_name', how = 'left')
df = geopandas.GeoDataFrame(df, geometry="geometry")
df['geometry'] = df['geometry'].centroid
df.rename(columns = {'geometry' : 'mun_geometry'}, inplace = True)
df['mun_lat'] = df.mun_geometry.y
df['mun_lon'] = df.mun_geometry.x

##now coordinates for US states
gdf = geopandas.read_file("c:\\data\\geo\\us-major-cities\\USA_Major_Cities.shp")
gdf = gdf[['NAME', 'geometry']].copy()
gdf.sort_values('NAME', inplace = True)
gdf.rename(columns = {'NAME' : 'registration_consulate'}, inplace = True)

miss = list(set(df.registration_consulate) - set(gdf.registration_consulate))
fix = ['San Jose', 'Del Rio', 'Indianapolis', 'Calexico',
       'Philadelphia', 'Boise City', 'Minneapolis',
       'New Orleans', 'New York', 'Los Angeles', 'McAllen',
       'Presidio']
dict_names = dict(zip(miss, fix))

df.loc[df.registration_consulate.isin(miss), 'registration_consulate'] = (
    df.loc[df.registration_consulate.isin(miss), 'registration_consulate'].map(dict_names))

df = df.merge(gdf[['registration_consulate', 'geometry']], on='registration_consulate', how = 'left')
df = geopandas.GeoDataFrame(df, geometry="geometry")
df['geometry'] = df['geometry'].centroid
df.rename(columns = {'geometry' : 'consul_geometry'}, inplace = True)

#fill in by hand the coordinates for Presidio
df.loc[df.registration_consulate == 'Presidio', 'consul_geometry'] = [geopandas.points_from_xy(x=[-104.368], y=[29.563])]

df['consul_lat'] = df.consul_geometry.y
df['consul_lon'] = df.consul_geometry.x

## fill in geo info for mexican state
gdf = geopandas.read_file("c:\\data\\geo\\world_admin2\\World_Administrative_Divisions.shp")
gdf = gdf[(gdf.COUNTRY == "Mexico") & (gdf.LAND_RANK == 5)][['NAME', 'geometry']]
gdf.sort_values('NAME', inplace = True)
gdf.rename(columns = {'NAME' : 'state'}, inplace = True)

miss = list(set(gdf.state)-set(df["Estado de Origen"]))
fix = ['Coahuila', 'Veracruz', 'Michoacán', 'Estado de México']
dict_names = dict(zip(miss, fix))
gdf.loc[gdf.state.isin(miss), 'state'] = (
    gdf.loc[gdf.state.isin(miss), 'state'].map(dict_names))

df = df.merge(gdf[['state', 'geometry']], left_on='Estado de Origen',
              right_on = 'state', how = 'left')
df = geopandas.GeoDataFrame(df, geometry="geometry")
df['geometry'] = df['geometry'].centroid
df.rename(columns = {'geometry' : 'state_geometry'}, inplace = True)

df['state_lat'] = df.state_geometry.y
df['state_lon'] = df.state_geometry.x

df.isna().sum()

df.to_csv(os.getcwd() + "\\mexico\\data\\consulates_network_with_geo.csv", index = False)