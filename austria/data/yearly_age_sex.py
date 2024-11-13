import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np

folder = "C:\\users\\andrea vismara\\downloads\\age_sex\\"

files = os.listdir(folder)

df = pd.DataFrame()
for file in tqdm(files):
    df_ = pd.read_excel(folder + file, skiprows=10, skipfooter=9).iloc[1:, 1:]
    df_.replace('-', 0, inplace =True)
    df_.columns = ['year', 'sex', 'age_group'] + [x for x in df_.columns[3:]]
    df_.ffill(inplace = True)
    df_ = pd.melt(df_, id_vars=['year', 'sex', 'age_group'], value_vars=[x for x in df_.columns[3:]],
                  var_name='country', value_name='people')
    df_ = df_.replace('up to 4 years old', '0 to 4 years old')
    df_['age_group'] = df_['age_group'].apply(lambda x: [int(s) for s in re.findall(r'\b\d+\b', x)])
    df_['mean_age'] = df_['age_group'].apply(np.mean)
    df = pd.concat([df, df_])

df.to_excel("c:\\data\\population\\austria\\age_sex_all_clean.xlsx", index = False)