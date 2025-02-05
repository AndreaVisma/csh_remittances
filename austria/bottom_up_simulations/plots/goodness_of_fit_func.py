import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

outfolder = ".\\austria\\plots\\plots_for_paper\\model_results\\"

def goodness_of_fit_results(df, pred = False):

    df['error'] = df['obs_remittances'] - df['sim_remittances']
    df['absolute_error'] = np.abs(df['error'])

    #  MAE
    MAE = df['absolute_error'].mean()
    print(f"Mean absolute error: {round(MAE, 3)}")

    # MSE
    MSE = np.mean(np.square(df['error']))
    print(f"Mean squared error: {round(MSE, 3)}")

    # RMSE
    RMSE = np.sqrt(MSE)
    print(f"Root mean squared error: {round(RMSE, 3)}")

    # R-squared
    SS_res = np.sum(np.square(df['error']))
    SS_tot = np.sum(np.square(df['obs_remittances'] - np.mean(df['obs_remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared: {round(R_squared, 3)}")

    # Calculate MAPE
    # Ensure no division by zero
    MAPE = (np.abs(df['error']).sum() / df['obs_remittances'].sum()) * 100
    print(f"Mean absolute percentage error: {round(MAPE, 3)}%")

    # Scatter Plot
    if pred:
        p_color = 'C1'
        title = 'Observed vs simulated remittances, prediction sample (2020-2023)'
    else:
        p_color = 'C0'
        title = 'Observed vs simulated remittances, training sample (2013-2019)'
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df['obs_remittances'], df['sim_remittances'], alpha=0.6, color = p_color)
    plt.xlabel('Observed Remittances')
    plt.ylabel('Simulated Remittances')
    plt.title(title)
    # Add identity line
    lims = [0, df['obs_remittances'].max()]
    ax.plot(lims, lims, 'k-', alpha=1, zorder=1)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    if pred:
        fig.savefig(outfolder + "prediction_results.pdf")
    else:
        fig.savefig(outfolder + "training_results.pdf")
    plt.show(block = True)
