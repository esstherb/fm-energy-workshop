import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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

# Split
feature_cols = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
test_hours = 14 * 24

train = df.iloc[:-test_hours]
test  = df.iloc[-test_hours:]

X_train = train[feature_cols]
y_train = train["electricity_kwh"]
X_test  = test[feature_cols]
y_test  = test["electricity_kwh"]

# Train
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("=== Training complete ===")
print(f"Model type      : {type(model).__name__}")
print(f"Trees built     : {model.n_estimators_}")
print(f"Features used   : {feature_cols}")

print("\n=== Feature importances ===")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in importances.items():
    print(f"  {feat:<12} {score:.4f}")

print("\n=== Train-set score (R²) ===")
print(f"  {model.score(X_train, y_train):.4f}  (1.0 = perfect fit on training data)")
