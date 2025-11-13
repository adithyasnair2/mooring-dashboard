import streamlit as st
import pandas as pd
import json
import glob
import os
import joblib

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
        return "🔧 Hardware fault - inspect immediately"
    if pd.isna(tension):
        return "ℹ️ No line attached"
    if tension >= 8:
        return f"⚠️ URGENT: Loosen {line_type} by 30–40% immediately"
    elif tension >= 6:
        return f"⚡ Loosen {line_type} by 20–30%"
    elif tension >= 5:
        return f"👀 Monitor closely. Consider loosening by 10–15%"
    elif tension <= 1:
        return f"💡 Tighten {line_type} by 20–30% for better hold"
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

# ---------- MODEL LOADER & FEATURE BUILDER ----------

@st.cache_resource
def load_model():
    """Load trained RandomForest model + feature list from disk."""
    try:
        model = joblib.load("mooring_risk_model.joblib")
        with open("mooring_risk_features.json", "r") as f:
            feature_names = json.load(f)
        return model, feature_names
    except Exception as e:
        # If model files aren't there, return None and handle in UI
        return None, None

def build_feature_row(num_crit, ship_dist, active_hooks, total_hooks, feature_names):
    """
    Build a single-row DataFrame of features for the model,
    using user inputs + reasonable defaults for other features.
    """
    # Start with zeros for all features
    row = {name: 0.0 for name in feature_names}

    # Basic guards
    total_hooks = max(total_hooks, 1)
    active_hooks = max(min(active_hooks, total_hooks), 0)
    num_crit = max(min(num_crit, active_hooks), 0)
    ship_dist = max(ship_dist, 0.0)

    util_ratio = active_hooks / total_hooks
    crit_ratio = num_crit / total_hooks
    # Estimate number of "high tension" lines (>=5): some + all critical
    est_high = max(num_crit, int(active_hooks * 0.3))
    high_ratio = est_high / total_hooks

    # Fill in features if they exist in this model
    if "total_hooks" in row:
        row["total_hooks"] = float(total_hooks)
    if "active_hooks" in row:
        row["active_hooks"] = float(active_hooks)
    if "util_ratio" in row:
        row["util_ratio"] = float(util_ratio)
    if "crit_ratio" in row:
        row["crit_ratio"] = float(crit_ratio)
    if "high_ratio" in row:
        row["high_ratio"] = float(high_ratio)

    if "num_tension_ge_7" in row:
        row["num_tension_ge_7"] = float(num_crit)
    if "num_tension_ge_5" in row:
        row["num_tension_ge_5"] = float(est_high)

    # Distances – assume this snapshot distance is representative
    if "min_ship_distance" in row:
        row["min_ship_distance"] = float(ship_dist)
    if "max_ship_distance" in row:
        row["max_ship_distance"] = float(ship_dist)
    if "mean_ship_distance" in row:
        row["mean_ship_distance"] = float(ship_dist)

    # Radar-based features – use reasonable defaults
    if "num_active_radars" in row:
        row["num_active_radars"] = 3.0
    if "max_distance_change" in row:
        row["max_distance_change"] = 3.0
    if "mean_distance_change" in row:
        row["mean_distance_change"] = 1.5

    # Faults – assume ~5% of hooks are faulted unless overridden
    est_faults = int(round(total_hooks * 0.05))
    if "faulted_hooks" in row:
        row["faulted_hooks"] = float(est_faults)
    if "fault_ratio" in row:
        row["fault_ratio"] = float(est_faults / total_hooks)

    # Tension statistics – scaled by how critical the system looks
    if "avg_tension" in row:
        row["avg_tension"] = 4.0 + 2.0 * crit_ratio
    if "max_tension" in row:
        row["max_tension"] = 8.0 if num_crit > 0 else 6.0
    if "std_tension" in row:
        row["std_tension"] = 1.5 + crit_ratio

    # Convert to DataFrame with correct column ordering
    X = pd.DataFrame([row], columns=feature_names)
    return X

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
model, feature_names = load_model()

# ---------- Sidebar (shared) ----------

st.sidebar.title("⚓ Controls")
st.sidebar.caption(f"Using latest snapshot: `{os.path.basename(latest_snapshot)}`")

ships = sorted(df_hooks["ship_name"].dropna().unique())
selected_ship = st.sidebar.selectbox("🚢 Select Ship", ships, index=0)

st.sidebar.markdown("### 🎯 Risk Filter (Live View)")
risk_filter = st.sidebar.radio(
    "Show hooks:",
    ["ALL", "FAULTED", "CRITICAL", "ALERT", "SAFE"],
    index=0
)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 5s – live tab only)", value=False)
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# ---------- Tabs ----------

tab_live, tab_history, tab_ai = st.tabs([
    "📡 Live Snapshot",
    "🕒 History (All Snapshots)",
    "🤖 AI Risk Predictor"
])

# ===========================
# TAB 1: LIVE SNAPSHOT
# ===========================
with tab_live:
    latest_id = os.path.basename(latest_snapshot)
    df_live = df_hooks[(df_hooks["snapshot_id"] == latest_id) &
                       (df_hooks["ship_name"] == selected_ship)].copy()

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
            - 🟧 **ALERT**: Tension 5–7
            - 🟡 **SLACK**: Tension ≤ 1
            - 🟢 **SAFE**: Tension 2–4
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
    st.caption("📡 Live snapshot from latest JSON | ⚓ Mooring Safety System v2.1")

# ===========================
# TAB 2: HISTORY VIEWER
# ===========================
with tab_history:
    st.title("🕒 Mooring History – All Snapshots")

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

# ===========================
# TAB 3: AI RISK PREDICTOR
# ===========================
with tab_ai:
    st.title("🤖 AI Mooring Risk Predictor")

    if model is None or feature_names is None:
        st.warning("Model files not found (`mooring_risk_model.joblib` & `mooring_risk_features.json`). "
                   "Place them in this folder to enable AI predictions.")
    else:
        st.markdown("Design a **what-if scenario** and let the model estimate the **probability of a high-risk mooring state**.")

        col_input1, col_input2 = st.columns(2)

        with col_input1:
            total_hooks_input = st.number_input(
                "Total hooks available for this ship",
                min_value=1,
                max_value=200,
                value=70,
                step=1
            )
            active_hooks_input = st.number_input(
                "Active hooks currently in use",
                min_value=0,
                max_value=int(total_hooks_input),
                value=65,
                step=1
            )

        with col_input2:
            critical_hooks_input = st.number_input(
                "Number of hooks in CRITICAL tension (≥ 8)",
                min_value=0,
                max_value=int(active_hooks_input),
                value=10,
                step=1
            )
            ship_distance_input = st.number_input(
                "Ship distance from berth (m)",
                min_value=0.0,
                max_value=50.0,
                value=7.0,
                step=0.5
            )

        st.markdown("---")

        if st.button("🔮 Run AI Risk Assessment"):
            X = build_feature_row(
                num_crit=critical_hooks_input,
                ship_dist=ship_distance_input,
                active_hooks=active_hooks_input,
                total_hooks=total_hooks_input,
                feature_names=feature_names
            )

            prob_high = float(model.predict_proba(X)[0][1])
            pred_class = int(model.predict(X)[0])

            st.subheader("📊 Model Output")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="High-Risk Probability",
                    value=f"{prob_high * 100:.1f} %"
                )
            with col_b:
                class_label = "HIGH RISK" if pred_class == 1 else "ACCEPTABLE / SAFE"
                class_emoji = "🚨" if pred_class == 1 else "✅"
                st.metric(
                    label="Predicted Class",
                    value=f"{class_emoji} {class_label}"
                )

            # Simple stakeholder explanation
            st.markdown("### 📝 Interpretation for stakeholders")
            if pred_class == 1:
                st.error(
                    "The model flags this mooring configuration as **HIGH RISK**.\n\n"
                    f"- **Critical hooks:** {critical_hooks_input} out of {active_hooks_input} active\n"
                    f"- **Utilisation:** {active_hooks_input}/{total_hooks_input} hooks in use\n"
                    f"- **Ship distance:** ~{ship_distance_input} m from berth\n\n"
                    "This combination suggests **very high tension concentration** on a subset of lines. "
                    "The crew should consider redistributing lines, easing some critical hooks, or adding extra lines "
                    "before proceeding with cargo operations."
                )
            else:
                st.success(
                    "The model classifies this configuration as **ACCEPTABLE / SAFE**, but not risk-free.\n\n"
                    f"- **Critical hooks:** {critical_hooks_input} out of {active_hooks_input} active\n"
                    f"- **Utilisation:** {active_hooks_input}/{total_hooks_input} hooks in use\n"
                    f"- **Ship distance:** ~{ship_distance_input} m from berth\n\n"
                    "From a mooring safety perspective, this looks like a **manageable loading pattern**. "
                    "Operators should still continue routine monitoring, particularly if weather or cargo conditions change."
                )

        st.caption("Model based on historical synthetic mooring data – for decision support, not a certified safety system.")

