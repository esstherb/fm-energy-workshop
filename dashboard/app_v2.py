"""
Energy Monitoring Dashboard  –  v2
Simulated live feed from meter_readings_source (Neon PostgreSQL).

Start-up  : loads all of January 2016 (2016-01-01 00:00 → 2016-01-31 23:00).
Every real minute : fetches the next recorded hour from Neon (one row per
                    meter) and appends it to the display.
Restart   : resets to January data and restarts the one-minute timer.
End       : stops refreshing once the last record (2016-02-29 23:00) has been
            retrieved.
"""

import os
import re
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Dashboard v2 – Live Simulation",
    page_icon="⚡",
    layout="wide",
)

# ── Neon connection ───────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
_raw_url = os.environ["DATABASE_URL"]
DATABASE_URL = re.sub(r"[&?]channel_binding=require", "", _raw_url).strip().strip('"')
_host = re.search(r"@([^/]+)/", DATABASE_URL).group(1)
_SQL_URL = f"https://{_host}/sql"
_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": DATABASE_URL,
}

COLORS = {
    "METER_01": "#3B82F6",   # blue
    "METER_02": "#10B981",   # emerald
    "METER_03": "#F59E0B",   # amber
    "Total":    "#8B5CF6",   # violet
}

# Simulation boundaries (UTC-aware)
SIM_START = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")  # first displayed hour
SIM_SEED_END = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")  # end of January seed
SIM_END = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")    # last record in DB
TICK_SECONDS = 60  # one real minute per simulated hour

# ── Neon query helper ─────────────────────────────────────────────────────────
def run_sql(query: str, params: list | None = None) -> list[dict]:
    payload: dict = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(_SQL_URL, json=payload, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["rows"]


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading January seed data …")
def load_january() -> pd.DataFrame:
    """Load all readings for 1–31 January 2016 from Neon."""
    rows = run_sql(
        """
        SELECT meter_id, timestamp, building_id, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp >= $1
          AND  timestamp <= $2
        ORDER  BY timestamp, meter_id
        """,
        ["2016-01-01 00:00:00+00", "2016-01-31 23:00:00+00"],
    )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


def fetch_next_hour(after: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch the single next distinct timestamp after `after` for all meters.
    Returns an empty DataFrame when no more rows exist.
    """
    rows = run_sql(
        """
        WITH next_ts AS (
            SELECT MIN(timestamp) AS ts
            FROM   meter_readings_source
            WHERE  timestamp > $1
        )
        SELECT m.meter_id, m.timestamp, m.building_id, m.electricity_kwh
        FROM   meter_readings_source m
        JOIN   next_ts n ON m.timestamp = n.ts
        ORDER  BY m.meter_id
        """,
        [after.isoformat()],
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state():
    """Populate session state from the January seed (runs once per session)."""
    seed = load_january()
    st.session_state.df = seed.copy()
    st.session_state.last_ts = seed["timestamp"].max()   # 2016-01-31 23:00
    st.session_state.sim_done = False
    st.session_state.last_tick = time.time()             # wall-clock of last fetch


if "df" not in st.session_state:
    _init_state()


# ── Restart button ────────────────────────────────────────────────────────────
col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("⚡ Energy Dashboard — Live Simulation")
with col_btn:
    st.write("")   # vertical spacing
    if st.button("🔄 Restart", use_container_width=True):
        load_january.clear()          # bust the cache so seed re-fetches cleanly
        _init_state()
        st.rerun()


# ── Tick logic: fetch next hour if a minute has elapsed ──────────────────────
if not st.session_state.sim_done:
    now = time.time()
    elapsed = now - st.session_state.last_tick

    if elapsed >= TICK_SECONDS:
        with st.spinner("Fetching next hour from Neon …"):
            new_rows = fetch_next_hour(st.session_state.last_ts)

        if new_rows.empty:
            st.session_state.sim_done = True
        else:
            new_ts = new_rows["timestamp"].iloc[0]
            if new_ts > SIM_END:
                st.session_state.sim_done = True
            else:
                st.session_state.df = pd.concat(
                    [st.session_state.df, new_rows], ignore_index=True
                )
                st.session_state.last_ts = new_ts
                st.session_state.last_tick = now


# ── Build pivot from current display data ────────────────────────────────────
df: pd.DataFrame = st.session_state.df
meters = sorted(df["meter_id"].unique())

pivot = df.pivot_table(
    index="timestamp", columns="meter_id",
    values="electricity_kwh", aggfunc="sum",
)
pivot["Total"] = pivot[list(meters)].sum(axis=1)
pivot = pivot.reset_index()


# ── Chart helpers ─────────────────────────────────────────────────────────────
def meter_label(mid: str) -> str:
    return mid.replace("_", " ").title()


def make_meter_chart(meter_id: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot["timestamp"],
        y=pivot[meter_id],
        mode="lines",
        line=dict(color=COLORS.get(meter_id, "#6B7280"), width=1.4),
        name=meter_label(meter_id),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=meter_label(meter_id), font_size=15, x=0.02),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(gridcolor="#E5E7EB", title="kWh"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def make_total_chart() -> go.Figure:
    fig = go.Figure()
    # Shaded area
    fig.add_trace(go.Scatter(
        x=pivot["timestamp"],
        y=pivot["Total"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["Total"], width=0),
        fillcolor="rgba(139,92,246,0.15)",
        name="Total",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Solid line on top
    fig.add_trace(go.Scatter(
        x=pivot["timestamp"],
        y=pivot["Total"],
        mode="lines",
        line=dict(color=COLORS["Total"], width=2),
        name="Total",
        showlegend=False,
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Total — All Meters Combined", font_size=15, x=0.02),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        hovermode="x unified",
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(gridcolor="#E5E7EB", title="kWh"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


# ── Status bar ────────────────────────────────────────────────────────────────
displayed_ts = st.session_state.last_ts
date_min = df["timestamp"].min().strftime("%d %b %Y %H:%M")
date_max = displayed_ts.strftime("%d %b %Y %H:%M")
total_kwh = df["electricity_kwh"].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Meters", len(meters))
m2.metric("Showing up to", date_max)
m3.metric("Readings displayed", f"{len(df):,}")
m4.metric("Cumulative kWh", f"{total_kwh:,.0f}")

# Simulation status badge
if st.session_state.sim_done:
    st.success("✅ Simulation complete — all records retrieved (Jan–Feb 2016).")
else:
    remaining_hours = int(
        (SIM_END - st.session_state.last_ts).total_seconds() / 3600
    )
    next_tick_in = max(0, TICK_SECONDS - (time.time() - st.session_state.last_tick))
    st.info(
        f"🟢 **Simulating live feed** — "
        f"{remaining_hours} hour(s) remaining · "
        f"next fetch in **{next_tick_in:.0f}s**",
        icon="📡",
    )

st.caption(
    "📂 Data source: **historical** (simulated live) — "
    "`meter_readings_source`, Neon PostgreSQL · "
    "1 real minute = 1 recorded hour"
)
st.divider()


# ── Individual meter charts ───────────────────────────────────────────────────
st.subheader("Hourly Electricity Usage — Individual Meters")
cols = st.columns(3)
for col, mid in zip(cols, meters):
    col.plotly_chart(make_meter_chart(mid), use_container_width=True)


# ── Total chart ───────────────────────────────────────────────────────────────
st.subheader("Hourly Electricity Usage — All Meters Combined")
st.plotly_chart(make_total_chart(), use_container_width=True)


st.divider()


# ── Placeholders ─────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("🔮 Prediction")
    st.info(
        "Energy usage forecasting will appear here.\n\n"
        "_Coming soon — this feature is not yet implemented._",
        icon="🚧",
    )

with right:
    st.subheader("🤖 LLM Analysis")
    st.info(
        "AI-powered energy analysis and Q&A will appear here.\n\n"
        "_Coming soon — this feature is not yet implemented._",
        icon="🚧",
    )


# ── Auto-rerun while simulation is running ────────────────────────────────────
# Schedule the next rerun to fire just after the next tick is due.
if not st.session_state.sim_done:
    wait_ms = max(1000, int((TICK_SECONDS - (time.time() - st.session_state.last_tick)) * 1000))
    st.markdown(
        f"<meta http-equiv='refresh' content='{wait_ms // 1000}'>",
        unsafe_allow_html=True,
    )


# ── Run instructions ──────────────────────────────────────────────────────────
st.divider()
st.markdown("**To run version 2 from the project root:**")
st.code(
    "vibe\\Scripts\\streamlit run dashboard\\app_v2.py --server.headless true",
    language="bash",
)
