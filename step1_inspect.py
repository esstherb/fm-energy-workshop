import pandas as pd

# Load the data
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])

# Basic shape and types
print("=== Shape ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")

print("\n=== Column types ===")
print(df.dtypes)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Summary statistics ===")
print(df["electricity_kwh"].describe().round(2))

# Date range
print("\n=== Date range ===")
print(f"From : {df['timestamp'].min()}")
print(f"To   : {df['timestamp'].max()}")
total_hours = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600) + 1
print(f"Expected hours : {total_hours}")
print(f"Actual rows    : {len(df)}")
print(f"Missing hours  : {total_hours - len(df)}")

# Missing values
print("\n=== Missing values ===")
print(df.isnull().sum())

# Duplicate timestamps
dupes = df["timestamp"].duplicated().sum()
print(f"\n=== Duplicate timestamps: {dupes} ===")

# Unique meters / buildings
print("\n=== Unique meter IDs ===")
print(df["meter_id"].unique())
print("\n=== Unique building IDs ===")
print(df["building_id"].unique())
