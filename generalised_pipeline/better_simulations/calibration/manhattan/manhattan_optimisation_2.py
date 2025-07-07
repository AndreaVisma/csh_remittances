
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from italy.simulation.func.goodness_of_fit import goodness_of_fit_results

with open('model_results_0406_all_params.pkl', 'rb') as fi:
    loaded_data = pickle.load(fi)

df_runs = pd.DataFrame([
    {**params, 'error': error, 'error_nomex' : error_nomex}
    for params, error, error_nomex in loaded_data
])
df_runs = df_runs[['nta', 'asy', 'gdp', 'height', 'shape', 'constant', 'error', 'error_nomex']]
df_runs.to_excel("simulations_errors.xlsx")

best = df_runs[df_runs.error == df_runs.error.min()]
param_nta = best['nta'].item()
param_stay = 0
param_asy = best['asy'].item()
param_gdp = best['gdp'].item()
height = best['height'].item()
shape =  best['shape'].item()
constant = best['constant'].item()
##################

####### now fit with regression
from sklearn.linear_model import LinearRegression

# twelve feature columns
df = df_runs.copy()
df['nta_quad'] = df['nta'] ** 2
df['asy_quad'] = df['asy'] ** 2
df['gdp_quad'] = df['gdp'] ** 2
df['const_quad'] = df['constant'] ** 2
df['height_quad'] = df['height'] ** 2
df['shape_quad'] = df['shape'] ** 2

feature_cols = [
    'nta', 'nta_quad',
    'asy', 'asy_quad',
    'gdp', 'gdp_quad',
    'constant', 'const_quad',
    'height', 'height_quad',
    'shape', 'shape_quad'
]

X = df[feature_cols].values       # 500×12 design matrix
y = df['error'].values            # 500×1 target vector

# Fit linear regression
reg = LinearRegression(fit_intercept=False)  # no extra intercept since 'constant' is a feature
reg.fit(X, y)

learned = reg.coef_
for name, coef in zip(feature_cols, learned):
    print(f"{name:10s} → {coef:.5f}")

df['error_est'] = reg.predict(X)
total_ss_err = ((df['error'] - df['error_est'])**2).sum()
print("Total SSE after fitting:", total_ss_err)

fig, ax = plt.subplots(figsize = (9,6))
ax.scatter(df['error'], df['error_est'])
plt.xlabel('Error from simulation run')
plt.ylabel('Error obtained by fitting parabolic function')
plt.plot([0, 6e16], [0, 6e16], '--', color='black', alpha=0.8)
plt.grid(True)
plt.show(block = True)

#################
# Extract each pair of coefficients:
coefs = dict(zip(feature_cols, reg.coef_))

alpha_1 = coefs['nta']
alpha_2 = coefs['nta_quad']
beta_1  = coefs['asy']
beta_2  = coefs['asy_quad']
chi_1   = coefs['gdp']
chi_2   = coefs['gdp_quad']
delta_1 = coefs['constant']
delta_2 = coefs['const_quad']
eta_1   = coefs['height']
eta_2   = coefs['height_quad']
phi_1   = coefs['shape']
phi_2   = coefs['shape_quad']

# minimizers, assuming, each quadratic coefficient > 0:
nta_opt      = -alpha_1  / (2 * alpha_2) ## but this has a negative quadratic coefficient, thats a problem
asy_opt      = -beta_1   / (2 * beta_2)
gdp_opt      = -chi_1    / (2 * chi_2) ## but this has a negative quadratic coefficient, thats a problem
constant_opt = -delta_1  / (2 * delta_2)
height_opt   = -eta_1    / (2 * eta_2)
shape_opt    = -phi_1    / (2 * phi_2)

print("Optimal nta     =", nta_opt)
print("Optimal asy     =", asy_opt)
print("Optimal gdp     =", gdp_opt)
print("Optimal constant=", constant_opt)
print("Optimal height  =", height_opt)
print("Optimal shape   =", shape_opt)

param_nta = nta_opt
param_stay = 0
param_asy = asy_opt
param_gdp = gdp_opt
height = height_opt
shape =  shape_opt
constant = constant_opt

#####################

rem_per_period = check_initial_guess_with_disasters(height, shape, fixed_probability=False, plot = True)