
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import griddata

df_gdp = pd.read_excel("c:\\data\\economic\\gdp\\annual_gdp_per_capita_clean.xlsx").groupby('country').mean().reset_index().rename(columns = {'country' : 'origin'}).drop(columns = 'year')

# df1 = pd.read_pickle('all_results_1806_1.pkl')
# df2 = pd.read_pickle('all_results_2306.pkl')
# df3 = pd.read_pickle('all_results_2406.pkl')
# df = pd.concat([df2, df3])

##############
df = pd.read_pickle('all_results_3006.pkl')

param_names = ['nta','asy','gdp','height','shape','constant']
df.loc[~df[param_names].duplicated(), 'run_nr'] = [x for x in range(500)]
df['run_nr'].ffill(inplace = True)
df['error'] = (df['sim_remittances'] - df['remittances']) / 1e9
df['error_rel_obs'] = np.abs(df['sim_remittances'] - df['remittances']) / (df['remittances'] + 1)
# df = df[df['error_rel_obs'] < 10]

df_sum = df[['run_nr', 'error', 'error_rel_obs']].groupby('run_nr').agg(
    mean_avg_err = pd.NamedAgg(column="error", aggfunc=lambda x: np.mean(np.abs(x))),
    mean_squared_err = pd.NamedAgg(column="error", aggfunc=lambda x: np.mean(np.square(x))),
    root_mean_squared_err = pd.NamedAgg(column="error", aggfunc=lambda x: np.sqrt(np.mean(np.square(x)))),
    sum_squared_err = pd.NamedAgg(column="error", aggfunc=lambda x: np.sum(np.square(x))),
    sum_squared_rel_err = pd.NamedAgg(column="error_rel_obs", aggfunc=lambda x: np.sum(np.square(x)))
).reset_index()

df_sum = df_sum.merge(df[~df.run_nr.duplicated()][['run_nr'] + param_names], how = 'left')
df_sum.sort_values('sum_squared_err', inplace = True)
df_sum.to_excel("C://Users//Andrea Vismara//Downloads//sim_errors_3006_nopct.xlsx", index = False)

##############

# parameter columns
param_cols = ["nta", "asy", "gdp", "height", "shape", "constant"]

# choose the error metric you want to model:
target_col = "sum_squared_err"

X = df_sum[param_cols].values        # shape (n_samples, n_params)
y = df_sum[target_col].values        # shape (n_samples,)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# hold out some data to check performance if you like
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

# initialize & train
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=0,
)
rf.fit(X_train, y_train)

# (optional) quick check
print("R² on held‐out data:", rf.score(X_test, y_test))


def error_surrogate(nta, asy, gdp, height, shape, constant):
    """
    Predicts the sum_squared_err for given parameters.
    Returns a scalar float.
    """
    xp = np.array([[nta, asy, gdp, height, shape, constant]])
    return float(rf.predict(xp))

# example usage:
print(error_surrogate(2.5, -7.1,  9.2, 0.3, 0.1,  -0.5))

from scipy.optimize import minimize

# initial guess (e.g., midpoints or best of your grid)
x0 = [df_sum[pc].mean() for pc in param_cols]

# bounds if you have them
bounds = [(df_sum[c].min(), df_sum[c].max()) for c in param_cols]

res = minimize(
    lambda x: error_surrogate(*x),
    x0,
    method="Nelder-Mead",
)
dict_best = dict(zip(param_cols, res.x))
for k, v in dict_best.items():
    print(f"{k}:{v}")
print("Predicted error:", res.fun)


def bootstrap_parameter_cis(X, y, bounds, x0,
                            n_boot=100, rf_kwargs=None,
                            opt_method="L-BFGS-B"):
    """
    Returns:
      theta_hats:   array shape (n_boot, n_params) of fitted params
      cis:          array shape (n_params, 2) of [2.5%, 97.5%] CIs
    """
    if rf_kwargs is None:
        rf_kwargs = dict(n_estimators=200, max_depth=10, random_state=0)

    n, p = X.shape
    theta_hats = np.zeros((n_boot, p))

    for b in tqdm(range(n_boot)):
        # 1) bootstrap sample indices
        idx = np.random.choice(n, size=int(0.6*n), replace=True)
        Xb, yb = X[idx], y[idx]

        # 2) fit RF on boot sample
        rf = RandomForestRegressor(**rf_kwargs)
        rf.fit(Xb, yb)

        # 3) define surrogate
        def surrogate(x):
            return float(rf.predict(x.reshape(1, -1)))

        # 4) optimize
        res = minimize(surrogate, x0, bounds=bounds, method=opt_method)
        theta_hats[b, :] = res.x

    # 5) compute CIs
    cis = np.percentile(theta_hats, [2.5, 97.5], axis=0).T
    return theta_hats, cis

# --- usage ---
theta_boot, theta_cis = bootstrap_parameter_cis(
    X, y, bounds=bounds, x0=x0, n_boot=100, opt_method="Nelder-Mead"
)

for i, pc in enumerate(param_cols):
    lo, hi = theta_cis[i]
    print(f"{pc:>8s} 95% CI: [{lo:.3f}, {hi:.3f}]")

##############
# Gaussian process

# 1) Gaussian Process surrogate with uncertainty estimates
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split

# assume X (n_samples×n_params) and y (n_samples,) are already defined
# e.g. from pandas.read_excel as in the earlier example

# split for quick validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# define a kernel: constant * RBF
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2))

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,           # noise level (can tune)
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=0
)

# fit the GP
gp.fit(X_train, y_train)

# evaluate
r2 = gp.score(X_test, y_test)
print(f"GP R² on held‐out data: {r2:.3f}")

# a callable that returns mean+std
def gp_surrogate(params):
    """
    params: array‐like of shape (n_params,)
    returns: (mean_pred, std_pred)
    """
    xp = np.atleast_2d(params)
    mu, std = gp.predict(xp, return_std=True)
    return float(mu), float(std)

# example
mean_pred, std_pred = gp_surrogate([2.5, -7.1,  9.2, 0.3, 0.1, 1, -0.5])
print("Predicted error:", mean_pred, "±", std_pred)

res = minimize(
    lambda x: gp_surrogate(x)[0],
    x0,
    method="L-BFGS-B",
)
print(param_cols)
print("Best‐fit parameters:", res.x)
print("Predicted error:", res.fun)

######
