import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import os
os.environ["PATH"] += os.pathsep + "C://Program Files//Graphviz//bin//"


df = pd.read_excel("c:\\data\\my_datasets\\remittances_austria_panel_quarterly.xlsx")
df = pd.get_dummies(df, columns=['year','quarter', 'group'], drop_first=True)
df.dropna(inplace=True)

X = df.drop(columns=['remittances', 'date', 'growth_rate_rem', 'country'])  # Drop date if it's not useful for predictions
y = df['remittances']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100
)

#optimal apparently
xg_reg = xgb.XGBRegressor(random_state=42,
                          alpha = 0, colsample_bytree = 0.7,
                         learning_rate = 0.2, max_depth = 5, n_estimators = 300,
                          subsample = 0.8)

xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_test)
y_pred_train = xg_reg.predict(X_train)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

xgb.plot_importance(xg_reg)
plt.show(block = True)

## test predictions
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(y_test, y_pred)
ax.plot(np.linspace(0, max(y_test), 140), np.linspace(0, max(y_test), 140), color='red')
plt.xlabel('observed remittances')
plt.ylabel('simulated remittances')
plt.title('Test values v. predicted test values')
plt.grid()
plt.show(block=True)

##train predictions
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(y_train, y_pred_train)
ax.plot(np.linspace(0, max(y_train), 140), np.linspace(0, max(y_train), 140), color='red')
plt.xlabel('observed remittances')
plt.ylabel('simulated remittances')
plt.title('Train values v. predicted training values')
plt.grid()
plt.show(block=True)

##hyperparameters tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'alpha': [0, 0.1, 1],
    'lambda': [1, 1.5, 2]
}

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
                          alpha = 0, colsample_bytree = 0.7,
                         learning_rate = 0.2, max_depth = 5, n_estimators = 300,
                          subsample = 0.8)