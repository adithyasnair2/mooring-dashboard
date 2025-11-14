import streamlit as st
import pandas as pd
import json
import glob
import os
import random

# Page config
st.set_page_config(page_title="Mooring Monitor", layout="wide")

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- Helper functions ----------

def classify_risk(row):
    """Classify hook risk level based on tension and fault status"""
    if row["hook_faulted"]:
        return "FAULTED"
    if pd.isna(row["hook_tension"]):
        return "NO_LINE"
    t = row["hook_tension"]
    if t >= 8:
        return "CRITICAL"
    elif t >= 5:
        return "ALERT"
    elif t <= 1:
        return "SLACK"
    else:
        return "SAFE"

def get_recommendation(row):
    """Generate action recommendation for crew"""
    tension = row["hook_tension"]
    faulted = row["hook_faulted"]
    line_type = row["hook_line_type"] if pd.notna(row["hook_line_type"]) else "line"
    
    if faulted:
        return "identified as faulty"
    if pd.isna(tension):
        return "ℹ️ No line attached"
    if tension >= 8:
        return f"⚠️ URGENT: Loosen {line_type} by 30-40% immediately"
    elif tension >= 6:
        return f"⚡ Loosen {line_type} by 20-30%"
    elif tension >= 5:
        return f"👀 Monitor closely. Consider loosening by 10-15%"
    elif tension <= 1:
        return f"💡 Tighten {line_type} by 20-30% for better hold"
    else:
        return "✅ Tension optimal"

def parse_snapshot_time(filename: str):
    """Extract timestamp from filename mooring_sample_YYYYMMDDHHMMSS.json"""
    base = os.path.basename(filename)
    ts_part = base.replace("mooring_sample_", "").replace(".json", "")
    try:
        return pd.to_datetime(ts_part, format="%Y%m%d%H%M%S")
    except Exception:
        return pd.NaT

def flatten_snapshot(json_obj, snapshot_id=None, snapshot_time=None):
    """Flatten a single port snapshot JSON into hook-level rows."""
    rows = []
    port_name = json_obj.get("name")

    for berth in json_obj.get("berths", []):
        berth_name = berth.get("name")
        ship = berth.get("ship", {}) or {}
        ship_name = ship.get("name")
        vessel_id = ship.get("vesselId") or ship.get("vessel_id")

        for bollard in berth.get("bollards", []):
            bollard_name = bollard.get("name")

            for hook in bollard.get("hooks", []):
                rows.append({
                    "snapshot_id": snapshot_id,
                    "snapshot_time": snapshot_time,
                    "port": port_name,
                    "berth": berth_name,
                    "ship_name": ship_name,
                    "vessel_id": vessel_id,
                    "bollard": bollard_name,
                    "hook_name": hook.get("name"),
                    "hook_tension": hook.get("tension"),
                    "hook_faulted": hook.get("faulted", False),
                    "hook_line_type": hook.get("attachedLine"),
                })
    return rows

# ---------- Load ALL JSON snapshots ----------

@st.cache_data
def load_all_data():
    """
    Load all mooring_sample_*.json files.
    Returns:
        df_hooks      - hook-level data for all snapshots
        df_history    - ship-level summary per snapshot
        latest_file   - filename of latest snapshot
    """
    files = sorted(glob.glob("mooring_sample_*.json"))
    if not files:
        st.error("❌ No mooring_sample_*.json files found in this folder.")
        st.stop()

    all_rows = []

    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
        snap_id = os.path.basename(fpath)
        snap_time = parse_snapshot_time(fpath)
        rows = flatten_snapshot(data, snapshot_id=snap_id, snapshot_time=snap_time)
        all_rows.extend(rows)

    if not all_rows:
        st.error("❌ No hook data found in JSON snapshots.")
        st.stop()

    df_hooks = pd.DataFrame(all_rows)

    # Clean dtypes
    df_hooks["hook_tension"] = pd.to_numeric(df_hooks["hook_tension"], errors="coerce")
    df_hooks["hook_faulted"] = df_hooks["hook_faulted"].astype(bool)
    df_hooks["snapshot_time"] = pd.to_datetime(df_hooks["snapshot_time"])

    # Risk classification + recommendations
    df_hooks["risk_level"] = df_hooks.apply(classify_risk, axis=1)
    df_hooks["recommendation"] = df_hooks.apply(get_recommendation, axis=1)

    risk_priority = {
        "FAULTED": 4,
        "CRITICAL": 3,
        "ALERT": 2,
        "SLACK": 1,
        "SAFE": 1,
        "NO_LINE": 0,
    }
    df_hooks["priority"] = df_hooks["risk_level"].map(risk_priority)

    # --- Build ship-level history summary ---
    group_cols = [
        "snapshot_id",
        "snapshot_time",
        "port",
        "berth",
        "ship_name",
        "vessel_id",
    ]

    def count_state(s, state):
        return (s == state).sum()

    df_history = (
        df_hooks
        .groupby(group_cols)
        .agg(
            total_hooks=("hook_name", "count"),
            active_hooks=("hook_tension", lambda s: s.notna().sum()),
            faulted_hooks=("hook_faulted", "sum"),
            critical_hooks=("risk_level", lambda s: count_state(s, "CRITICAL")),
            alert_hooks=("risk_level", lambda s: count_state(s, "ALERT")),
            slack_hooks=("risk_level", lambda s: count_state(s, "SLACK")),
            safe_hooks=("risk_level", lambda s: count_state(s, "SAFE")),
            no_line_hooks=("risk_level", lambda s: count_state(s, "NO_LINE")),
            avg_tension=("hook_tension", "mean"),
            max_tension=("hook_tension", "max"),
        )
        .reset_index()
        .sort_values("snapshot_time")
    )

    latest_file = files[-1]
    return df_hooks, df_history, latest_file

df_hooks, df_history, latest_snapshot = load_all_data()

# ---------- Sidebar (shared) ----------

st.sidebar.title("⚓ Controls")

ships = sorted(df_hooks["ship_name"].dropna().unique())
selected_ship = st.sidebar.selectbox("🚢 Select Ship", ships, index=0)

# NEW: pick RANDOM snapshot for this ship 
ship_snaps = df_hooks.loc[df_hooks["ship_name"] == selected_ship, "snapshot_id"].unique()

if len(ship_snaps) == 0:
    chosen_snapshot = None
    st.sidebar.caption("No snapshots available for this ship.")
else:
    chosen_snapshot = random.choice(list(ship_snaps))
    st.sidebar.caption(f"Random snapshot selected: `{chosen_snapshot}`")

st.sidebar.markdown("### 🎯 Risk Filter (Live View)")
risk_filter = st.sidebar.radio(
    "Show hooks:",
    ["ALL", "FAULTED", "CRITICAL", "ALERT", "SAFE"],
    index=0
)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 5s - live tab only)", value=False)
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# ---------- Tabs (2 only) ----------

tab_live, tab_history = st.tabs([
    "📡 Live Snapshot",
    "🕒 History (All Snapshots)",
])

# TAB 1: LIVE SNAPSHOT
with tab_live:
    if chosen_snapshot is None:
        st.info("No snapshot available for this ship.")
        st.stop()

    df_live = df_hooks[
        (df_hooks["snapshot_id"] == chosen_snapshot) &
        (df_hooks["ship_name"] == selected_ship)
    ].copy()

    st.title("⚓ Real-Time Mooring Hook Monitor")
    if "vessel_id" in df_live.columns and len(df_live) > 0:
        st.markdown(f"**Ship:** {selected_ship} | **Vessel ID:** {df_live['vessel_id'].iloc[0]}")
    else:
        st.markdown(f"**Ship:** {selected_ship}")

    total_hooks = len(df_live)
    critical_hooks = (df_live["risk_level"] == "CRITICAL").sum()
    alert_hooks = (df_live["risk_level"] == "ALERT").sum()
    faulted_hooks = (df_live["risk_level"] == "FAULTED").sum()
    safe_hooks = (df_live["risk_level"] == "SAFE").sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Total Hooks", total_hooks)
    with col2:
        st.metric("🔴 Critical", critical_hooks,
                  delta=f"-{critical_hooks}" if critical_hooks > 0 else "0",
                  delta_color="inverse")
    with col3:
        st.metric("🟧 Alert", alert_hooks,
                  delta=f"-{alert_hooks}" if alert_hooks > 0 else "0",
                  delta_color="inverse")
    with col4:
        st.metric("⚠️ Faulted", faulted_hooks,
                  delta=f"-{faulted_hooks}" if faulted_hooks > 0 else "0",
                  delta_color="inverse")
    with col5:
        st.metric("✅ Safe", safe_hooks,
                  delta=f"+{safe_hooks}" if safe_hooks > 0 else "0",
                  delta_color="normal")

    if critical_hooks > 0 or faulted_hooks > 0:
        st.error(f"⚠️ **ATTENTION REQUIRED:** {critical_hooks + faulted_hooks} hooks need immediate action!")

    df_view = df_live.copy()
    if risk_filter != "ALL":
        df_view = df_view[df_view["risk_level"] == risk_filter]

    df_view = df_view.sort_values(["priority", "hook_tension"], ascending=[False, False])

    st.markdown("---")
    st.subheader("📋 Hook Status & Recommendations")
    st.caption(f"Showing **{len(df_view)}** of **{total_hooks}** hooks")

    if len(df_view) > 0:
        def color_risk(val):
            colors = {
                "FAULTED": "background-color: #dc2626; color: white; font-weight: bold",
                "CRITICAL": "background-color: #ef4444; color: white; font-weight: bold",
                "ALERT": "background-color: #fb923c; color: black",
                "SLACK": "background-color: #fde047; color: black",
                "SAFE": "background-color: #86efac; color: black",
                "NO_LINE": "background-color: #d1d5db; color: #6b7280",
            }
            return colors.get(val, "")

        display_df = df_view[[
            "berth",
            "bollard",
            "hook_name",
            "hook_line_type",
            "hook_tension",
            "risk_level",
            "recommendation",
        ]].rename(columns={
            "berth": "Berth",
            "bollard": "Bollard",
            "hook_name": "Hook",
            "hook_line_type": "Line Type",
            "hook_tension": "Tension",
            "risk_level": "Status",
            "recommendation": "⚡ Action Required",
        })

        st.dataframe(
            display_df.style.applymap(color_risk, subset=["Status"]),
            use_container_width=True,
            height=500,
        )
    else:
        st.info("ℹ️ No hooks found matching the selected filter")

    st.markdown("---")
    with st.expander("ℹ️ **Risk Level Guide & Thresholds**"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Risk Levels:**
            - 🔴 **FAULTED**: Hardware fault detected
            - 🔴 **CRITICAL**: Tension ≥ 8
            - 🟧 **ALERT**: Tension 5-7
            - 🟡 **SLACK**: Tension ≤ 1
            - 🟢 **SAFE**: Tension 2-4
            - ⚪ **NO_LINE**: No line attached
            """)
        with col2:
            st.markdown("""
            **Line Types:**
            - **HEAD**: Forward mooring lines
            - **STERN**: Aft mooring lines
            - **SPRING**: Fore/aft spring lines
            - **BREAST**: Lateral mooring lines
            """)

    st.markdown("---")
    st.caption("📡 Live snapshot from random JSON | ⚓ Mooring Safety System v2.1")

# TAB 2: HISTORY VIEWER
with tab_history:
    st.title("🕒 Mooring History - All Snapshots")

    df_hist_ship = df_history[df_history["ship_name"] == selected_ship].copy()

    if df_hist_ship.empty:
        st.info("No historical data found for this ship.")
    else:
        df_hist_ship = df_hist_ship.sort_values("snapshot_time")

        st.subheader(f"Snapshot Summary for **{selected_ship}**")
        st.caption("Each row = one JSON snapshot for this ship")

        st.dataframe(
            df_hist_ship[[
                "snapshot_time",
                "berth",
                "total_hooks",
                "active_hooks",
                "critical_hooks",
                "alert_hooks",
                "faulted_hooks",
                "safe_hooks",
                "avg_tension",
                "max_tension",
            ]].rename(columns={
                "snapshot_time": "Timestamp",
                "berth": "Berth",
                "total_hooks": "Total Hooks",
                "active_hooks": "Active Hooks",
                "critical_hooks": "Critical",
                "alert_hooks": "Alert",
                "faulted_hooks": "Faulted",
                "safe_hooks": "Safe",
                "avg_tension": "Avg Tension",
                "max_tension": "Max Tension",
            }),
            use_container_width=True,
            height=350,
        )

        st.markdown("---")
        st.subheader("📈 Risk Trend Over Time")

        trend_df = df_hist_ship.set_index("snapshot_time")[
            ["critical_hooks", "alert_hooks", "faulted_hooks", "safe_hooks"]
        ]
        st.line_chart(trend_df)

        st.markdown("---")
        st.subheader("⚙️ Tension Trend Over Time")

        tension_df = df_hist_ship.set_index("snapshot_time")[
            ["avg_tension", "max_tension"]
        ]
        st.line_chart(tension_df)

        st.caption("Each point represents one JSON snapshot in time for this ship.")
