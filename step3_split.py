import pandas as pd

# Rebuild the featured dataset
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

# Hold back the last 2 weeks (336 hours) as the test set
test_hours = 14 * 24  # 336

train = df.iloc[:-test_hours].copy()
test  = df.iloc[-test_hours:].copy()

print("=== Split summary ===")
print(f"Total usable rows : {len(df)}")
print(f"Train rows        : {len(train)}")
print(f"Test rows         : {len(test)}")

print("\n=== Train date range ===")
print(f"From : {train['timestamp'].min()}")
print(f"To   : {train['timestamp'].max()}")

print("\n=== Test date range ===")
print(f"From : {test['timestamp'].min()}")
print(f"To   : {test['timestamp'].max()}")

print("\n=== No overlap check ===")
overlap = set(train["timestamp"]) & set(test["timestamp"])
print(f"Overlapping timestamps : {len(overlap)}  (should be 0)")
