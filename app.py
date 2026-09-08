"""
Streamlit Dashboard — Smart Home Security MVP
-----------------------------------------------
The "output screen" of the pipeline: talks to the FastAPI backend over
HTTP, never touches the database or the model directly (except to
generate synthetic events client-side before sending them in).

Run locally (with the backend already running on port 8000):
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import requests
import streamlit as st

from simulator import generate_events

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Smart Home Security Dashboard", layout="wide", page_icon="🏠")

RISK_COLORS = {"low": "#2ecc71", "medium": "#f1c40f", "high": "#e67e22", "critical": "#e74c3c"}


# ------------------------------------------------------------------
# API helpers — every call is defensive: the backend may not be running.
# ------------------------------------------------------------------
def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach backend at `{API_URL}{path}` — is FastAPI running? ({e})")
        return None


def api_post(path: str, json_body=None):
    try:
        r = requests.post(f"{API_URL}{path}", json=json_body, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach backend at `{API_URL}{path}` — is FastAPI running? ({e})")
        return None


def api_patch(path: str):
    try:
        r = requests.patch(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None


# ------------------------------------------------------------------
# Sidebar — pipeline controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    API_URL = st.text_input("Backend API URL", value=API_URL)

    st.subheader("1️⃣ Simulate events")
    sim_days = st.slider("Days of history to generate", 1, 14, 3)
    sim_anomaly_rate = st.slider("Anomaly rate", 0.0, 0.2, 0.05, step=0.01)
    if st.button("🎲 Generate & send events", use_container_width=True):
        events = generate_events(
            datetime.utcnow() - timedelta(days=sim_days),
            num_days=sim_days,
            anomaly_rate=sim_anomaly_rate,
            seed=None,  # fresh randomness each click
        )
        # Strip the debug-only ground-truth field before sending — a real
        # device would never report whether its own event "is" an anomaly.
        payload = [{k: v for k, v in e.items() if k != "_injected_anomaly"}
                   for e in events]

        progress = st.progress(0.0, text=f"Sending {len(payload)} events...")
        chunk_size = 300
        for i in range(0, len(payload), chunk_size):
            chunk = payload[i:i + chunk_size]
            result = api_post("/events/bulk", chunk)
            if result is None:
                break
            progress.progress(min((i + chunk_size) / len(payload), 1.0),
                               text=f"Sent {min(i + chunk_size, len(payload))}/{len(payload)}")
        progress.empty()
        st.success(f"Sent {len(payload)} simulated events.")
        st.rerun()

    st.subheader("2️⃣ Run AI analysis")
    st.caption("Scores every event not yet analyzed.")
    if st.button("🧠 Analyze pending events", use_container_width=True):
        result = api_post("/analyze", {})
        if result is not None:
            st.success(f"Analyzed {len(result)} event(s).")
            st.rerun()

    st.divider()
    if st.button("🔄 Refresh dashboard", use_container_width=True):
        st.rerun()


# ------------------------------------------------------------------
# Pipeline overview banner
# ------------------------------------------------------------------
st.title("🏠 Smart Home Security Dashboard")

pipeline_stages = [
    ("📡", "Simulated Events", "Door, motion, network, energy"),
    ("🔌", "FastAPI Backend", "Collects & validates"),
    ("🧹", "Data Processing", "Time, frequency, device features"),
    ("🌲", "AI Anomaly Engine", "Isolation Forest"),
    ("⚠️", "Risk Layer", "Rules + context"),
    ("🗄️", "Database", "Events, scores, alerts"),
    ("📊", "Dashboard", "You are here"),
]
cols = st.columns(len(pipeline_stages))
for col, (icon, name, desc) in zip(cols, pipeline_stages):
    with col:
        st.markdown(
            f"""<div style="text-align:center; padding:10px; border-radius:8px;
                 background:#f0f2f6;">
                 <div style="font-size:28px;">{icon}</div>
                 <div style="font-weight:600; font-size:13px;">{name}</div>
                 <div style="font-size:11px; color:#666;">{desc}</div>
                 </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ------------------------------------------------------------------
# Pull data
# ------------------------------------------------------------------
all_events = api_get("/events", {"limit": 500}) or []
scored_events = api_get("/events/scored", {"limit": 500}) or []
alerts = api_get("/alerts", {"limit": 200}) or []

df_scored = pd.DataFrame(scored_events)
df_alerts = pd.DataFrame(alerts)

total_events = len(all_events)
analyzed_count = sum(1 for e in all_events if e.get("analyzed"))
anomaly_count = int(df_scored["is_anomaly"].sum()) if not df_scored.empty and "is_anomaly" in df_scored else 0
active_alerts = int((~df_alerts["acknowledged"]).sum()) if not df_alerts.empty else 0
critical_alerts = int((df_alerts["risk_level"] == "critical").sum()) if not df_alerts.empty else 0

# ------------------------------------------------------------------
# Status metrics
# ------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Events", total_events)
m2.metric("Analyzed", analyzed_count, delta=f"{total_events - analyzed_count} pending"
          if total_events - analyzed_count else None)
m3.metric("Anomalies Detected", anomaly_count)
m4.metric("Active Alerts", active_alerts)
m5.metric("Critical Alerts", critical_alerts)

st.divider()

# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Risk Level Distribution")
    if not df_scored.empty and "risk_level" in df_scored and df_scored["risk_level"].notna().any():
        dist = (df_scored.dropna(subset=["risk_level"])
                .groupby("risk_level").size().reset_index(name="count"))
        order = ["low", "medium", "high", "critical"]
        chart = alt.Chart(dist).mark_bar().encode(
            x=alt.X("risk_level:N", sort=order, title="Risk Level"),
            y=alt.Y("count:Q", title="Events"),
            color=alt.Color("risk_level:N",
                             scale=alt.Scale(domain=order,
                                             range=[RISK_COLORS[r] for r in order]),
                             legend=None),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No analyzed events yet — generate events and run analysis.")

with chart_col2:
    st.subheader("Anomaly Score Over Time")
    if not df_scored.empty and "anomaly_score" in df_scored and df_scored["anomaly_score"].notna().any():
        plot_df = df_scored.dropna(subset=["anomaly_score", "risk_level"]).copy()
        plot_df["timestamp"] = pd.to_datetime(
    plot_df["timestamp"],
    format="ISO8601",
    errors="coerce"
)

# Remove rows where timestamp couldn't be parsed
        plot_df = plot_df.dropna(subset=["timestamp"])
        order = ["low", "medium", "high", "critical"]
        chart = alt.Chart(plot_df).mark_circle(size=60).encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("anomaly_score:Q", title="Anomaly Score"),
            color=alt.Color("risk_level:N",
                             scale=alt.Scale(domain=order,
                                             range=[RISK_COLORS[r] for r in order])),
            tooltip=["timestamp:T", "device_id:N", "event_type:N",
                     "anomaly_score:Q", "risk_level:N"],
        ).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No analyzed events yet — generate events and run analysis.")

st.divider()

# ------------------------------------------------------------------
# Recent events table
# ------------------------------------------------------------------
st.subheader("Recent Events")
if not df_scored.empty:
    display_cols = ["timestamp", "device_id", "event_type", "anomaly_score", "risk_level"]
    display_df = df_scored[display_cols].copy()

    # ✅ FIX: robust timestamp parsing
    display_df["timestamp"] = pd.to_datetime(
        display_df["timestamp"], format="ISO8601", errors="coerce"
    )
    display_df = display_df.dropna(subset=["timestamp"])
    display_df = display_df.sort_values("timestamp", ascending=False)

    def _risk_style(val):
        color = RISK_COLORS.get(val, "#ffffff")
        return f"background-color: {color}33; color: {color}; font-weight: 600;"

    st.dataframe(
        display_df.style.map(_risk_style, subset=["risk_level"]),
        use_container_width=True,
        height=350,
    )
else:
    st.info("No events yet. Use the sidebar to generate simulated events.")


# ------------------------------------------------------------------
# Alerts
# ------------------------------------------------------------------
st.subheader("🚨 Alerts")
if not df_alerts.empty:
    unresolved = df_alerts[~df_alerts["acknowledged"]].sort_values("created_at", ascending=False)
    resolved = df_alerts[df_alerts["acknowledged"]]

    if unresolved.empty:
        st.success("No active alerts.")
    for _, alert in unresolved.iterrows():
        color = RISK_COLORS.get(alert["risk_level"], "#999")
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f"""<div style="border-left: 4px solid {color}; padding: 8px 12px;
                     margin-bottom: 6px; background: {color}11;">
                     <b style="color:{color};">{alert['risk_level'].upper()}</b>
                     — {alert['message']}<br>
                     <span style="font-size:12px; color:#888;">{alert['created_at']}</span>
                     </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                api_patch(f"/alerts/{alert['id']}/ack")
                st.rerun()

    if not resolved.empty:
        with st.expander(f"Acknowledged alerts ({len(resolved)})"):
            st.dataframe(resolved[["created_at", "risk_level", "message"]],
                         use_container_width=True)
else:
    st.info("No alerts yet.")
