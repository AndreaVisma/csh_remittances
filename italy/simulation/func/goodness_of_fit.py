import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

outfolder = ".\\italy\\plots\\plots_for_paper\\model_results\\"

def goodness_of_fit_results(df, pred = False):

    df['error'] = df['remittances'] - df['sim_remittances']
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
    SS_tot = np.sum(np.square(df['remittances'] - np.mean(df['remittances'])))
    R_squared = 1 - (SS_res / SS_tot)
    print(f"R-squared: {round(R_squared, 3)}")

    # Calculate MAPE
    # Ensure no division by zero
    MAPE = (np.abs(df['error']).sum() / df['remittances'].sum()) * 100
    print(f"Mean absolute percentage error: {round(MAPE, 3)}%")

    # Scatter Plot
    if pred:
        p_color = 'C1'
        title = 'Observed vs simulated remittances, prediction sample (2020-2023)'
    else:
        p_color = 'C0'
        title = 'Observed vs simulated remittances, training sample (2013-2019)'
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df['remittances'], df['sim_remittances'], alpha=0.6, color = p_color)
    plt.xlabel('Observed Remittances')
    plt.ylabel('Simulated Remittances')
    plt.title(title)
    # Add identity line
    lims = [0, df['remittances'].max()]
    ax.plot(lims, lims, 'k-', alpha=1, zorder=1)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    if pred:
        fig.savefig(outfolder + "prediction_results.pdf")
    else:
        fig.savefig(outfolder + "training_results.pdf")
    plt.show(block = True)

def plot_lines(df):
    fig = go.Figure()

    # Trace for simulated_senders (using the left y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['simulated_senders'],
        name='Simulated Senders',
        mode='lines',
        marker=dict(color='blue')
    ))

    # Trace for population (using the left y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['population'],
        name='Population',
        mode='lines+markers',
        marker=dict(color='green')
    ))

    # Trace for remittances (using the right y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['remittances'],
        name='Remittances',
        mode='lines+markers',
        marker=dict(color='red'),
        yaxis='y2'
    ))

    # Update the layout to add a second y-axis
    fig.update_layout(
        title='Simulated Senders & Population vs Remittances Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Simulated Senders & Population',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis2=dict(
            title='Remittances',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )

    # Show the plot
    fig.show()

def plot_all_results_log(df):
    fig = px.scatter(df, x = 'remittances', y = 'sim_remittances',
                     color = 'country', log_x=True, log_y=True)
    fig.add_scatter(x=np.linspace(0, df.remittances.max(), 100),
                    y=np.linspace(0, df.remittances.max(), 100))
    fig.show()