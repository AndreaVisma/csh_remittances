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

df_tot[['year', 'Número de Matrículas']].groupby('year').sum()
df_unique_mex = pd.DataFrame.from_dict({"municipios" : df_tot['Municipio de Origen'].unique().tolist()})
df_unique_us = pd.DataFrame.from_dict({"consulate" : df_tot['registration_consulate'].unique().tolist()})
df_unique_mex.to_excel(os.getcwd() + "\\mexico\\data\\municipios.xlsx", index = False)
df_unique_us.to_excel(os.getcwd() + "\\mexico\\data\\consulates.xlsx", index = False)