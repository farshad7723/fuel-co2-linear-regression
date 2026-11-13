"""
Simple Linear Regression: predicting vehicle CO2 emissions.

Two models are trained and compared:
    1. Engine size (L)      -> CO2 emissions (g/km)
    2. Combined (L/100 km)  -> CO2 emissions (g/km)

The notebook simple_linear_regression.ipynb has the same steps with full explanations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = Path(__file__).parent / 'my2026-fuel-consumption-ratings.csv'
images_dir = Path(__file__).parent / 'images'
images_dir.mkdir(exist_ok=True)

# 1. Load the data
df = pd.read_csv(file_path)
print(df.shape)

# 2. Understand the data
print(df.columns.tolist())
print(df.isnull().sum())      # all 0 -> no missing values
print(df.describe())          # 'Model year' has std = 0 -> not useful

# 3. Explore the data
cdf = df[['Engine size (L)', 'Cylinders', 'Combined (L/100 km)', 'CO2 emissions (g/km)']]

# 3.1 Histograms
cdf.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig(images_dir / '01_histograms.png', dpi=120)
plt.show()

# 3.2 Scatter plots: which feature is most linear with CO2?
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

ax[0].scatter(cdf['Combined (L/100 km)'], cdf['CO2 emissions (g/km)'], color='blue')
ax[0].set_title('CO2 vs Fuel consumption')
ax[0].set_xlabel('Combined (L/100 km)')
ax[0].set_ylabel('CO2 emissions (g/km)')

ax[1].scatter(cdf['Engine size (L)'], cdf['CO2 emissions (g/km)'], color='green')
ax[1].set_title('CO2 vs Engine size')
ax[1].set_xlabel('Engine size (L)')

ax[2].scatter(cdf['Cylinders'], cdf['CO2 emissions (g/km)'], color='red')
ax[2].set_title('CO2 vs Cylinders')
ax[2].set_xlabel('Cylinders')

plt.tight_layout()
plt.savefig(images_dir / '02_scatter_comparison.png', dpi=120)
plt.show()

# 3.3 Correlation with CO2 (close to 1 = strong linear relationship)
print(cdf.corr()['CO2 emissions (g/km)'])

# 4. Split: 80% train, 20% test
#    Same random_state for both splits -> both models use exactly the same vehicles.
y = cdf['CO2 emissions (g/km)'].to_numpy()

X_eng = cdf['Engine size (L)'].to_numpy()
X_fuel = cdf['Combined (L/100 km)'].to_numpy()

X_eng_train, X_eng_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
X_fuel_train, X_fuel_test, y_train, y_test = train_test_split(X_fuel, y, test_size=0.2, random_state=42)

print('Train:', X_eng_train.shape)
print('Test: ', X_eng_test.shape)

# 5. Train the models
#    reshape(-1, 1): scikit-learn needs 2D input (rows = vehicles, columns = features)
model_eng = LinearRegression()
model_eng.fit(X_eng_train.reshape(-1, 1), y_train)

model_fuel = LinearRegression()
model_fuel.fit(X_fuel_train.reshape(-1, 1), y_train)

print('Model eng coef:', model_eng.coef_[0], 'Model eng intercept:', model_eng.intercept_)
print('Model fuel coef:', model_fuel.coef_[0], 'Model fuel intercept:', model_fuel.intercept_)

# 6. Fitted lines on the training data
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

ax[0].scatter(X_eng_train, y_train, color='blue')
ax[0].plot(X_eng_train, model_eng.coef_[0] * X_eng_train + model_eng.intercept_, color='red')
ax[0].set_title('Engine size model (train)')
ax[0].set_xlabel('Engine size (L)')
ax[0].set_ylabel('CO2 emissions (g/km)')

ax[1].scatter(X_fuel_train, y_train, color='blue')
ax[1].plot(X_fuel_train, model_fuel.coef_[0] * X_fuel_train + model_fuel.intercept_, color='red')
ax[1].set_title('Fuel consumption model (train)')
ax[1].set_xlabel('Combined (L/100 km)')

plt.tight_layout()
plt.savefig(images_dir / '03_fitted_lines_train.png', dpi=120)
plt.show()

# 7. Evaluate on the test data (vehicles the models have never seen)
y_pred_eng = model_eng.predict(X_eng_test.reshape(-1, 1))
y_pred_fuel = model_fuel.predict(X_fuel_test.reshape(-1, 1))

print("----- Engine size model -----")
print("MAE :  %.2f" % mean_absolute_error(y_test, y_pred_eng))
print("MSE :  %.2f" % mean_squared_error(y_test, y_pred_eng))
print("RMSE:  %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_eng)))
print("R2  :  %.3f" % r2_score(y_test, y_pred_eng))

print("----- Fuel consumption model -----")
print("MAE :  %.2f" % mean_absolute_error(y_test, y_pred_fuel))
print("MSE :  %.2f" % mean_squared_error(y_test, y_pred_fuel))
print("RMSE:  %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_fuel)))
print("R2  :  %.3f" % r2_score(y_test, y_pred_fuel))
