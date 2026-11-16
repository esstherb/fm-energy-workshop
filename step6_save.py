import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file without needing a display
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Rebuild dataset ──────────────────────────────────────────────────────────
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

# ── Train ────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
test["predicted"] = model.predict(X_test)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(test["timestamp"], test["electricity_kwh"], label="Actual",    color="steelblue",  linewidth=1.2)
ax.plot(test["timestamp"], test["predicted"],       label="Predicted", color="darkorange", linewidth=1.2, linestyle="--")
ax.set_title("Actual vs Predicted — two-week test window (16–29 Feb 2016)")
ax.set_xlabel("Date")
ax.set_ylabel("Electricity (kWh)")
ax.legend()
fig.tight_layout()

os.makedirs("models", exist_ok=True)
plot_path = "models/actual_vs_predicted.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Plot saved  : {plot_path}")

# ── Save model and feature order ─────────────────────────────────────────────
model_path   = "models/gbr_model.joblib"
feature_path = "models/feature_order.json"

joblib.dump(model, model_path)
print(f"Model saved : {model_path}")

with open(feature_path, "w") as f:
    json.dump(feature_cols, f)
print(f"Features saved : {feature_path}")

# ── Confirm files exist and sizes ─────────────────────────────────────────────
for path in [model_path, feature_path, plot_path]:
    size = os.path.getsize(path)
    print(f"  {path}  ({size:,} bytes)")

# ── Final metric reminder ────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, test["predicted"])
rmse = root_mean_squared_error(y_test, test["predicted"])
print(f"\nTest MAE  : {mae:.3f} kWh")
print(f"Test RMSE : {rmse:.3f} kWh")
