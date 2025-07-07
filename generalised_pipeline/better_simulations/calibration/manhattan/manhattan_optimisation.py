
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from italy.simulation.func.goodness_of_fit import goodness_of_fit_results

with open('model_results_2705_all_params.pkl', 'rb') as fi:
    loaded_data = pickle.load(fi)

df_runs = pd.DataFrame([
    {**params, 'error': error, 'error_nomex' : error_nomex}
    for params, error, error_nomex in loaded_data
])
df_runs = df_runs[['nta', 'asy', 'gdp', 'height', 'shape', 'error', 'error_nomex']]

# pd.plotting.scatter_matrix()
# plt.show(block = True)

plt.subplots()
plt.scatter(df_runs['nta'], df_runs['error'])
plt.show(block = True)

#### simply take the minimum error:
best = df_runs[df_runs.error == df_runs.error.min()]
param_nta = best['nta'].item()
param_stay = 0
param_asy = best['asy'].item()
param_gdp = best['gdp'].item()
height = best['height'].item()
shape = 0.5# best['shape'].item()

rem_per_period = check_initial_guess_with_disasters(height, shape, fixed_probability=False, plot = True)
rem_per_period = rem_per_period[rem_per_period.origin != "Mexico"]
goodness_of_fit_results(rem_per_period)
plot_country_mean(rem_per_period, two_countries=True)

##################
# optimise
###################

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from skopt import forest_minimize
from skopt.space import Real


# 1) Prepare X and y
df = df_runs.copy()
y = df['error'].values

# 2) Manually add squared terms
for feat in ['nta', 'asy', 'gdp', 'height', 'shape']:
    df[f'{feat}_2'] = df[feat] ** 2

X = df[['nta', 'asy', 'gdp', 'height', 'shape',
        'nta_2', 'asy_2', 'gdp_2', 'height_2', 'shape_2']].values

# 3) Build & fit pipeline
model = make_pipeline(
    StandardScaler(),      # scale all 10 features
    LinearRegression()     # y = β0 + Σ β_i x_i + Σ γ_i x_i^2
)
model.fit(X, y)

##############
# 1. Get predictions on your training (or test) set
# X = df_runs[['nta', 'asy', 'gdp', 'height', 'shape']].values
y_true = df['error'].values
y_pred = model.predict(X)

# 2. Scatter plot: True vs. Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6)
# plot 45° line
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, '--', linewidth=1, label='Ideal fit')
plt.xlabel('True Error')
plt.ylabel('Predicted Error')
plt.title('True vs. Predicted Errors')
plt.legend()
plt.tight_layout()
plt.show(block = True)

poly = model.named_steps['polynomialfeatures']
linreg = model.named_steps['linearregression']
feature_names = poly.get_feature_names_out(['nta','asy','gdp','height','shape'])
coefs = linreg.coef_

plt.figure(figsize=(8,4))
plt.bar(feature_names, coefs)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Coefficient value')
plt.title('Learned coefficients (linear and quadratic terms)')
plt.tight_layout()
plt.show(block = True)
#######################

# Define the same search space
space = [
    Real(0, 8.0, name='nta'),
    Real(-4, 0, name='asy'),
    Real(0, 8, name='gdp'),
    Real(0, 1, name='height'),
    Real(0, 2, name='shape')
]

# Objective function uses our fitted model to predict error
def error_function(params):
    x = np.array(params).reshape(1, -1)
    return model.predict(x)[0]

# Run Bayesian (forest) optimization
res = forest_minimize(error_function, space, n_calls=50, random_state=42)

print("Best parameters found:", dict(zip([s.name for s in space], res.x)))
print("Estimated minimum error:", res.fun)

#######################################

dict_res = dict(zip([s.name for s in space], res.x))
param_nta = dict_res['nta']
param_stay = 0
param_asy = dict_res['asy']
param_gdp = dict_res['gdp']
height = dict_res['height']
shape = dict_res['shape']

rem_per_period = check_initial_guess_with_disasters(height, shape, fixed_probability=False, plot = True)


#######################################
# plot errors surface
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Subset to needed columns
df_sub = df_runs[['height', 'shape', 'error']]

# Define grid
asy_vals = np.linspace(df_runs['height'].min(), df_runs['height'].max(), 100)
gdp_vals = np.linspace(df_runs['shape'].min(), df_runs['shape'].max(), 100)
asy_grid, gdp_grid = np.meshgrid(asy_vals, gdp_vals)

# Interpolate error values onto the grid
error_grid = griddata(
    points=df_sub[['height', 'shape']].values,
    values=df_sub['error'].values,
    xi=(asy_grid, gdp_grid),
    method='linear'  # use 'linear' or 'nearest' if you have sparse data
)

plt.figure(figsize=(10, 6))
contour = plt.contourf(asy_grid, gdp_grid, error_grid, levels=30, cmap='viridis')
plt.colorbar(contour, label='Error')
plt.xlabel('height')
plt.ylabel('shape')
plt.title('Error Surface: height vs shape')
plt.show(block = True)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(asy_grid, gdp_grid, error_grid, cmap='viridis')
ax.set_xlabel('height')
ax.set_ylabel('shape')
ax.set_zlabel('Error')
ax.set_title('3D Error Surface')
plt.show(block = True)

