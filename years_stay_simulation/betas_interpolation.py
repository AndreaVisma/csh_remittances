
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Original data (converted from comma to dot for floats)
growth_rate = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
beta = np.array([12.5, 17.5, 18.25, 18.5, 9.75, 6.25, 3.75, 2.75])

# Spline interpolation
cs = CubicSpline(growth_rate, beta)

# Estimate beta every 0.2 units in the growth_rate range
x_new = [round(x, 2) for x in np.arange(min(growth_rate), max(growth_rate)+0.01, 0.01)]
beta_estimated = cs(x_new)

# Optional: Plotting
plt.scatter(growth_rate, beta, color='red', label='Original Data')
plt.plot(x_new, beta_estimated, label='Cubic Spline')
plt.xlabel('Growth Rate')
plt.ylabel('Beta')
plt.title('Spline Interpolation of Beta vs Growth Rate')
plt.legend()
plt.grid(True)
plt.show(block = True)

df_betas = pd.DataFrame({'yrly_growth_rate' : x_new, 'beta_estimate' : beta_estimated})
df_betas.to_pickle("C:\\Data\\migration\\simulations\\exponential_betas.pkl")