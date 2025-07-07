
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import requests
from io import BytesIO
import country_converter as coco
cc = coco.CountryConverter()

# Function to get the flag image from the internet
def get_flag_image(country_code):
    url = f"https://flagcdn.com/w40/{country_code}.png"
    response = requests.get(url)
    if response.status_code == 200:
        return OffsetImage(plt.imread(BytesIO(response.content)), zoom=0.5)
    else:
        return None

def scatter_with_flags(df):
    # Create Scatter Plot
    df = df[['country', 'remittances', 'sim_remittances']].groupby(['country']).mean().reset_index()
    df['iso_code'] = cc.pandas_convert(series=df['country'], to='ISO3')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['remittances'], df['sim_remittances'], alpha=0)

    # Add flags to the scatter plot
    for i, iso_code in enumerate(df['iso_code']):
        flag_image = get_flag_image(iso_code)
        if flag_image:
            ab = AnnotationBbox(flag_image, (df['remittances'][i], df['sim_remittances'][i]), frameon=False)
            ax.add_artist(ab)

    # Labels and Title
    plt.xlabel('Actual Remittances')
    plt.ylabel('Simulated Remittances')
    plt.title('Remittances: Actual vs Simulated')
    plt.grid(True)
    plt.show(block = True)