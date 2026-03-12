import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np, json, os

df = pd.read_csv('data/housing.csv')
df = df.dropna()
df = pd.get_dummies(df, columns=['ocean_proximity'])

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
n    = len(X_train)

print(f'Training samples: {n}')
print(f'RMSE: {rmse:.2f}')
print(f'R2:   {r2:.4f}')

os.makedirs('metrics', exist_ok=True)
with open('metrics/metrics.json', 'w') as f:
    json.dump({'rmse': rmse, 'r2': r2, 'train_samples': n}, f, indent=2)
