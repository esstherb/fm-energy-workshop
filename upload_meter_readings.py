import csv
import os
import re
import requests

# ── Load DATABASE_URL from .env ──────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
database_url = None
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            database_url = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

if not database_url:
    raise RuntimeError("DATABASE_URL not found in .env")

# Strip channel_binding=require (unsupported by HTTP API)
database_url = re.sub(r"[&?]channel_binding=require", "", database_url)

# ── Derive the Neon HTTP SQL endpoint from the host in the connection string ──
# postgresql://user:pass@host/db?params  →  https://host/sql
match = re.search(r"@([^/]+)/", database_url)
if not match:
    raise RuntimeError("Cannot parse host from DATABASE_URL")
neon_host = match.group(1)
sql_url = f"https://{neon_host}/sql"

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": database_url,
}


def run_sql(query: str, params: list | None = None) -> dict:
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(sql_url, json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Create table ─────────────────────────────────────────────────────────────
print("Creating table meter_readings_source …")
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    meter_id        TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    building_id     TEXT        NOT NULL,
    electricity_kwh NUMERIC(12, 4) NOT NULL,
    PRIMARY KEY (meter_id, timestamp)
)
""")
print("Table ready.")

# ── Load CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "data", "three_meters.csv")
rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((
            row["meter_id"],
            row["timestamp"],
            row["building_id"],
            row["electricity_kwh"],
        ))

print(f"CSV rows loaded: {len(rows)}")

# ── Upload in batches using unnest ───────────────────────────────────────────
# Neon HTTP API accepts $1..$n positional params; we use unnest for bulk insert.
# Smaller batches keep each request well within timeout limits.
BATCH = 100
uploaded = 0

for start in range(0, len(rows), BATCH):
    batch = rows[start : start + BATCH]
    # Build parallel arrays for unnest
    meter_ids   = [r[0] for r in batch]
    timestamps  = [r[1] for r in batch]
    building_ids = [r[2] for r in batch]
    kwh_values  = [float(r[3]) for r in batch]

    run_sql(
        """
        INSERT INTO meter_readings_source (meter_id, timestamp, building_id, electricity_kwh)
        SELECT * FROM unnest($1::text[], $2::timestamptz[], $3::text[], $4::numeric[])
            AS t(meter_id, timestamp, building_id, electricity_kwh)
        ON CONFLICT (meter_id, timestamp) DO NOTHING
        """,
        [meter_ids, timestamps, building_ids, kwh_values],
    )
    uploaded += len(batch)
    print(f"  Uploaded {uploaded}/{len(rows)} rows …")

print(f"\nUpload complete. {uploaded} rows processed.")

# ── Summary query ─────────────────────────────────────────────────────────────
print("\n── Per-meter summary ──────────────────────────────────────────────────")
result = run_sql("""
SELECT
    meter_id,
    COUNT(*)                          AS row_count,
    MIN(timestamp) AT TIME ZONE 'UTC' AS earliest,
    MAX(timestamp) AT TIME ZONE 'UTC' AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

# Rows are dicts keyed by column name
print(f"{'meter_id':<12}  {'rows':>6}  {'earliest':<26}  {'latest':<26}")
print("-" * 76)
for row in result["rows"]:
    print(f"{row['meter_id']:<12}  {row['row_count']:>6}  {str(row['earliest']):<26}  {str(row['latest']):<26}")
