import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Rebuild featured dataset
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

feature_cols = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
test_hours = 14 * 24

train = df.iloc[:-test_hours]
test  = df.iloc[-test_hours:].copy()

X_train = train[feature_cols]
y_train = train["electricity_kwh"]
X_test  = test[feature_cols]
y_test  = test["electricity_kwh"]

# Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Model predictions
test["predicted"] = model.predict(X_test)

# Baseline: previous hour's reading (lag_1h)
test["baseline"] = test["lag_1h"]

# Metrics
model_mae  = mean_absolute_error(y_test, test["predicted"])
model_rmse = root_mean_squared_error(y_test, test["predicted"])

base_mae   = mean_absolute_error(y_test, test["baseline"])
base_rmse  = root_mean_squared_error(y_test, test["baseline"])

print("=== Test-set evaluation (336 hours = 2 weeks) ===")
print(f"\n{'Metric':<8}  {'Model':>10}  {'Baseline':>10}  {'Improvement':>12}")
print("-" * 46)
print(f"{'MAE':<8}  {model_mae:>9.3f}  {base_mae:>9.3f}  {(1 - model_mae/base_mae)*100:>10.1f}%")
print(f"{'RMSE':<8}  {model_rmse:>9.3f}  {base_rmse:>9.3f}  {(1 - model_rmse/base_rmse)*100:>10.1f}%")

print("\n=== Context: mean consumption in test period ===")
print(f"  Mean actual kWh : {y_test.mean():.2f}")
print(f"  Model MAE as % of mean : {model_mae / y_test.mean() * 100:.1f}%")

print("\n=== Sample predictions vs actuals (first 5 test rows) ===")
sample = test[["timestamp", "electricity_kwh", "predicted", "baseline"]].head(5)
sample.columns = ["timestamp", "actual", "model_pred", "baseline_pred"]
print(sample.to_string(index=False))
