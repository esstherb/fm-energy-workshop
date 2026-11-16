"""
Energy Monitoring Dashboard
Historical data from meter_readings_source (Neon PostgreSQL).
"""

import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
_raw_url = os.environ["DATABASE_URL"]
DATABASE_URL = re.sub(r"[&?]channel_binding=require", "", _raw_url).strip().strip('"')

_host = re.search(r"@([^/]+)/", DATABASE_URL).group(1)
_SQL_URL = f"https://{_host}/sql"
_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": DATABASE_URL,
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Loading historical readings …")
def load_data() -> pd.DataFrame:
    resp = requests.post(
        _SQL_URL,
        json={"query": "SELECT meter_id, timestamp, building_id, electricity_kwh FROM meter_readings_source ORDER BY timestamp"},
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame(payload["rows"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


df = load_data()
meters = sorted(df["meter_id"].unique())

# Pivot: one column per meter
pivot = df.pivot_table(index="timestamp", columns="meter_id", values="electricity_kwh", aggfunc="sum")
pivot["Total"] = pivot.sum(axis=1)
pivot = pivot.reset_index()

# ── Colour palette (one per meter + total) ────────────────────────────────────
COLORS = {
    meters[0]: "#3B82F6",   # blue
    meters[1]: "#10B981",   # emerald
    meters[2]: "#F59E0B",   # amber
    "Total":   "#8B5CF6",   # violet
}

def meter_label(mid: str) -> str:
    """Return 'METER_01' as 'Meter 01' for display."""
    return mid.replace("_", " ").title()

def make_meter_chart(meter_id: str) -> go.Figure:
    fig = px.line(
        pivot,
        x="timestamp",
        y=meter_id,
        labels={"timestamp": "", meter_id: "kWh"},
        color_discrete_sequence=[COLORS[meter_id]],
    )
    fig.update_traces(line_width=1.4)
    fig.update_layout(
        title=dict(text=meter_label(meter_id), font_size=15, x=0.02),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
        hovermode="x unified",
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(gridcolor="#E5E7EB", title="kWh"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig

def make_total_chart() -> go.Figure:
    fig = px.area(
        pivot,
        x="timestamp",
        y="Total",
        labels={"timestamp": "", "Total": "kWh"},
        color_discrete_sequence=[COLORS["Total"]],
    )
    fig.update_traces(line_width=1.6, opacity=0.25)
    # Overlay solid line
    fig.add_scatter(
        x=pivot["timestamp"],
        y=pivot["Total"],
        mode="lines",
        line=dict(color=COLORS["Total"], width=2),
        name="Total",
        showlegend=False,
    )
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

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Energy Monitoring Dashboard")

date_min = df["timestamp"].min().strftime("%d %b %Y")
date_max = df["timestamp"].max().strftime("%d %b %Y")
total_kwh = df["electricity_kwh"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Meters", len(meters))
c2.metric("Period", f"{date_min} – {date_max}")
c3.metric("Total readings", f"{len(df):,}")
c4.metric("Total consumption", f"{total_kwh:,.0f} kWh")

st.caption("📂 Data source: **historical** — `meter_readings_source` table, Neon PostgreSQL")
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

# ── Run instructions ──────────────────────────────────────────────────────────
st.divider()
st.markdown("**To run this dashboard from the project root:**")
st.code(
    "vibe\\Scripts\\streamlit run dashboard\\app.py --server.headless true",
    language="bash",
)
