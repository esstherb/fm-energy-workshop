"""
Energy Monitoring Dashboard  –  v3
Extends v2 with a Predict next hour button in the Prediction area.

Prediction uses the saved GradientBoostingRegressor (models/gbr_model.joblib)
and the saved feature order (models/feature_order.json).  The model is NOT
retrained; it is loaded once and reused.

Features required per prediction row:
    hour_of_day  – hour of the target timestamp (0-23)
    day_of_week  – weekday of the target timestamp (Mon=0, Sun=6)
    lag_1h       – kWh reading 1 h before target
    lag_2h       – kWh reading 2 h before target
    lag_3h       – kWh reading 3 h before target
    lag_24h      – kWh reading 24 h before target
    lag_168h     – kWh reading 168 h before target  (1 week)

Minimum history needed: 168 readings per meter.
If fewer are available, the prediction for that meter shows an explanation.
"""

import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Dashboard v3 – Prediction",
    page_icon="⚡",
    layout="wide",
)

# ── Neon connection ───────────────────────────────────────────────────────────
_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(_ROOT, "..", ".env"))
_raw_url = os.environ["DATABASE_URL"]
DATABASE_URL = re.sub(r"[&?]channel_binding=require", "", _raw_url).strip().strip('"')
_host = re.search(r"@([^/]+)/", DATABASE_URL).group(1)
_SQL_URL = f"https://{_host}/sql"
_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": DATABASE_URL,
}

COLORS = {
    "METER_01": "#3B82F6",
    "METER_02": "#10B981",
    "METER_03": "#F59E0B",
    "Total":    "#8B5CF6",
}

SIM_SEED_END = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")
SIM_END      = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")
TICK_SECONDS = 60

# ── Load model (once per process) ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path   = os.path.join(_ROOT, "..", "models", "gbr_model.joblib")
    feature_path = os.path.join(_ROOT, "..", "models", "feature_order.json")
    model = joblib.load(model_path)
    with open(feature_path) as f:
        features = json.load(f)
    return model, features


GBR_MODEL, FEATURE_ORDER = load_model()
MIN_HISTORY = 168   # lag_168h requires 168 prior readings

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


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_next_hour(meter_df: pd.DataFrame) -> dict:
    """
    Given all currently-displayed readings for one meter (sorted by timestamp),
    build the feature row for the NEXT hour and return a result dict:
        { "ok": True,  "timestamp": <pd.Timestamp>, "kwh": <float> }
        { "ok": False, "reason": <str> }
    """
    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)
    n = len(meter_df)

    if n < MIN_HISTORY:
        need = MIN_HISTORY - n
        return {
            "ok": False,
            "reason": (
                f"Need {MIN_HISTORY} readings for lag_168h; "
                f"only {n} available — {need} more hour(s) required."
            ),
        }

    kwh = meter_df["electricity_kwh"].values   # chronological array

    # Target timestamp = last known + 1 hour
    last_ts = meter_df["timestamp"].iloc[-1]
    target_ts = last_ts + pd.Timedelta(hours=1)

    row = {
        "hour_of_day": target_ts.hour,
        "day_of_week": target_ts.dayofweek,
        "lag_1h":      kwh[-1],
        "lag_2h":      kwh[-2],
        "lag_3h":      kwh[-3],
        "lag_24h":     kwh[-24],
        "lag_168h":    kwh[-168],
    }

    X = pd.DataFrame([row])[FEATURE_ORDER]
    pred = float(GBR_MODEL.predict(X)[0])
    pred = max(0.0, pred)   # clamp negatives

    return {"ok": True, "timestamp": target_ts, "kwh": pred}


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state():
    seed = load_january()
    st.session_state.df        = seed.copy()
    st.session_state.last_ts   = seed["timestamp"].max()
    st.session_state.sim_done  = False
    st.session_state.last_tick = time.time()
    st.session_state.pred_results = None   # holds last prediction output


if "df" not in st.session_state:
    _init_state()


# ── Header row: title + Restart ───────────────────────────────────────────────
col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("⚡ Energy Dashboard — Live Simulation")
with col_btn:
    st.write("")
    if st.button("🔄 Restart", width="stretch"):
        load_january.clear()
        _init_state()
        st.rerun()


# ── Tick logic ────────────────────────────────────────────────────────────────
if not st.session_state.sim_done:
    now = time.time()
    if now - st.session_state.last_tick >= TICK_SECONDS:
        with st.spinner("Fetching next hour from Neon …"):
            new_rows = fetch_next_hour(st.session_state.last_ts)
        if new_rows.empty or new_rows["timestamp"].iloc[0] > SIM_END:
            st.session_state.sim_done = True
        else:
            st.session_state.df = pd.concat(
                [st.session_state.df, new_rows], ignore_index=True
            )
            st.session_state.last_ts   = new_rows["timestamp"].iloc[0]
            st.session_state.last_tick = now
            # Invalidate previous prediction when new data arrives
            st.session_state.pred_results = None


# ── Build pivot ───────────────────────────────────────────────────────────────
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


def make_meter_chart(meter_id: str, pred_ts=None, pred_kwh=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot["timestamp"],
        y=pivot[meter_id],
        mode="lines",
        line=dict(color=COLORS.get(meter_id, "#6B7280"), width=1.4),
        name=meter_label(meter_id),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>",
    ))
    # Overlay prediction marker if available
    if pred_ts is not None and pred_kwh is not None:
        fig.add_trace(go.Scatter(
            x=[pred_ts],
            y=[pred_kwh],
            mode="markers+text",
            marker=dict(color="#EF4444", size=10, symbol="diamond"),
            text=[f"{pred_kwh:.1f}"],
            textposition="top center",
            textfont=dict(color="#EF4444", size=11),
            name="Predicted",
            hovertemplate="%{x|%d %b %H:%M}<br>Predicted: %{y:.2f} kWh<extra></extra>",
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
    fig.add_trace(go.Scatter(
        x=pivot["timestamp"], y=pivot["Total"],
        mode="lines", fill="tozeroy",
        line=dict(color=COLORS["Total"], width=0),
        fillcolor="rgba(139,92,246,0.15)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pivot["timestamp"], y=pivot["Total"],
        mode="lines",
        line=dict(color=COLORS["Total"], width=2),
        showlegend=False,
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Total — All Meters Combined", font_size=15, x=0.02),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300, hovermode="x unified",
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(gridcolor="#E5E7EB", title="kWh"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    )
    return fig


# ── Status bar ────────────────────────────────────────────────────────────────
date_max  = st.session_state.last_ts.strftime("%d %b %Y %H:%M")
total_kwh = df["electricity_kwh"].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Meters", len(meters))
m2.metric("Showing up to", date_max)
m3.metric("Readings displayed", f"{len(df):,}")
m4.metric("Cumulative kWh", f"{total_kwh:,.0f}")

if st.session_state.sim_done:
    st.success("✅ Simulation complete — all records retrieved (Jan–Feb 2016).")
else:
    remaining = int((SIM_END - st.session_state.last_ts).total_seconds() / 3600)
    next_in   = max(0, TICK_SECONDS - (time.time() - st.session_state.last_tick))
    st.info(
        f"🟢 **Simulating live feed** — "
        f"{remaining} hour(s) remaining · next fetch in **{next_in:.0f}s**",
        icon="📡",
    )

st.caption(
    "📂 Data source: **historical** (simulated live) — "
    "`meter_readings_source`, Neon PostgreSQL · "
    "1 real minute = 1 recorded hour"
)
st.divider()


# ── Individual meter charts ───────────────────────────────────────────────────
# Pull prediction results (if any) so charts can show the marker
preds = st.session_state.pred_results or {}

st.subheader("Hourly Electricity Usage — Individual Meters")
cols = st.columns(3)
for col, mid in zip(cols, meters):
    p = preds.get(mid, {})
    pts  = p.get("timestamp") if p.get("ok") else None
    pkwh = p.get("kwh")       if p.get("ok") else None
    col.plotly_chart(make_meter_chart(mid, pts, pkwh), width="stretch")


# ── Total chart ───────────────────────────────────────────────────────────────
st.subheader("Hourly Electricity Usage — All Meters Combined")
st.plotly_chart(make_total_chart(), width="stretch")

st.divider()


# ── Prediction + LLM placeholders ────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("🔮 Prediction")
    st.caption(
        "Model: GradientBoostingRegressor · "
        "`models/gbr_model.joblib` · "
        "features: hour_of_day, day_of_week, lag_1h/2h/3h/24h/168h"
    )

    if st.button("Predict next hour", type="primary", width="stretch"):
        results = {}
        for mid in meters:
            meter_df = df[df["meter_id"] == mid].copy()
            results[mid] = predict_next_hour(meter_df)
        st.session_state.pred_results = results

    # Show results if they exist
    if st.session_state.pred_results:
        st.markdown("---")
        for mid in meters:
            result = st.session_state.pred_results[mid]
            color  = COLORS.get(mid, "#6B7280")
            label  = meter_label(mid)

            if result["ok"]:
                target = result["timestamp"].strftime("%d %b %Y %H:%M UTC")
                kwh    = result["kwh"]
                st.markdown(
                    f"<div style='"
                    f"border-left: 4px solid {color};"
                    f"padding: 8px 12px; margin-bottom: 8px;"
                    f"background:#F9FAFB; border-radius:4px'>"
                    f"<b style='color:{color}'>{label}</b><br>"
                    f"<span style='font-size:1.4em;font-weight:700'>{kwh:.2f} kWh</span>"
                    f"<br><span style='color:#6B7280;font-size:0.85em'>Target: {target}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"**{label}** — {result['reason']}", icon="⚠️")
    else:
        st.info(
            "Press **Predict next hour** to run the model on current readings.",
            icon="👆",
        )

with right:
    st.subheader("🤖 LLM Analysis")
    st.info(
        "AI-powered energy analysis and Q&A will appear here.\n\n"
        "_Coming soon — this feature is not yet implemented._",
        icon="🚧",
    )


# ── Auto-rerun while simulation is running ────────────────────────────────────
if not st.session_state.sim_done:
    wait_s = max(1, int(TICK_SECONDS - (time.time() - st.session_state.last_tick)))
    st.markdown(
        f"<meta http-equiv='refresh' content='{wait_s}'>",
        unsafe_allow_html=True,
    )


# ── Run instructions ──────────────────────────────────────────────────────────
st.divider()
st.markdown("**To run version 3 from the project root:**")
st.code(
    "vibe\\Scripts\\streamlit run dashboard\\app_v3.py --server.headless true",
    language="bash",
)
