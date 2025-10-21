
import pandas as pd

emdat = pd.read_pickle("C:\\Data\\my_datasets\\monthly_disasters_with_lags_NEW.pkl")

emdat_f = emdat[emdat.date.dt.year == 2015]