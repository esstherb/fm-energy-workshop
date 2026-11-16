import pandas as pd

# Load and sort by time
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Time-of-day and day-of-week features
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0, Sunday=6

# Lag features — past readings only (no future data leakage)
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)

# Drop the first 168 rows that cannot have a full week of history
df = df.dropna().reset_index(drop=True)

print("=== Rows after dropping incomplete history ===")
print(f"{len(df)} rows remaining  (1440 - 168 = {1440 - 168} expected)")

print("\n=== Date range after drop ===")
print(f"From : {df['timestamp'].min()}")
print(f"To   : {df['timestamp'].max()}")

print("\n=== Feature columns ===")
feature_cols = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
print(feature_cols)

print("\n=== Sample rows (first 3) ===")
print(df[["timestamp", "electricity_kwh"] + feature_cols].head(3).to_string(index=False))

print("\n=== Missing values after feature build ===")
print(df[feature_cols].isnull().sum())
