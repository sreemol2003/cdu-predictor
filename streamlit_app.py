import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import hashlib
import requests
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- App Config ---
st.set_page_config(page_title="CDU Hybrid Digital Twin Platform", layout="wide")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DB_PATH = "audit_telemetry.db"
GUEST_MODEL_FILE = "models/guest_model.pkl"

# ==============================================================================
# DATABASE & ACCESS CONTROL LAYER (ADMIN & CLIENT PROVISIONING)
# ==============================================================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            role TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            login_time TIMESTAMP,
            ip_address TEXT,
            city TEXT,
            region TEXT,
            country TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS protected_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            model_tag TEXT,
            model_path TEXT,
            training_rows INTEGER,
            created_at TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS simulation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TIMESTAMP,
            inputs_json TEXT,
            outputs_json TEXT
        )
    ''')
    
    # Pre-seed default Admin and initial Engineer account
    admin_pw = hashlib.sha256("Admin@123".encode()).hexdigest()
    user_pw = hashlib.sha256("User@123".encode()).hexdigest()
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", ("admin", admin_pw, "admin"))
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", ("engineer1", user_pw, "user"))
    conn.commit()
    conn.close()

init_db()

def verify_login(username, password):
    clean_u = username.strip().lower()
    clean_p = password.strip()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", 
              (clean_u, hashlib.sha256(clean_p.encode()).hexdigest()))
    row = c.fetchone()
    conn.close()
    return row["role"] if row else None

def create_client_user(new_username, plain_password):
    clean_u = new_username.strip().lower()
    clean_p = plain_password.strip()
    p_hash = hashlib.sha256(clean_p.encode()).hexdigest()
    
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')", (clean_u, p_hash))
        conn.commit()
        success = True
        msg = f"Client account `{clean_u}` created successfully."
    except sqlite3.IntegrityError:
        success = False
        msg = f"Username `{clean_u}` already exists."
    conn.close()
    return success, msg

def reset_client_password(target_username, new_plain_password):
    p_hash = hashlib.sha256(new_plain_password.strip().encode()).hexdigest()
    conn = get_db_connection()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (p_hash, target_username))
    conn.commit()
    conn.close()

def delete_client_user(target_username):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE username = ?", (target_username,))
    conn.execute("DELETE FROM protected_models WHERE username = ?", (target_username,))
    conn.execute("DELETE FROM simulation_history WHERE username = ?", (target_username,))
    conn.commit()
    conn.close()

def get_visitor_geo():
    try:
        res = requests.get("https://ipapi.co/json/", timeout=2.5).json()
        return {
            "ip": res.get("ip", "Local/VPN"),
            "city": res.get("city", "Unknown"),
            "region": res.get("region", "Unknown"),
            "country": res.get("country_name", "Unknown")
        }
    except Exception:
        return {"ip": "127.0.0.1", "city": "Internal", "region": "Internal", "country": "Internal"}

def log_login_event(username):
    geo = get_visitor_geo()
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO access_logs (username, login_time, ip_address, city, region, country)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, datetime.now(), geo["ip"], geo["city"], geo["region"], geo["country"]))
    conn.commit()
    conn.close()

# ==============================================================================
# PHYSICS-INFORMED CONTINUOUS ENGINE
# ==============================================================================
YIELD_BOUNDS = {
    'flow_offgas': (0.005, 0.045),
    'flow_naphtha': (0.080, 0.320),
    'flow_kero': (0.050, 0.220),
    'flow_lago': (0.150, 0.420),
    'flow_residue': (0.220, 0.650)
}

PHYSICS_SLOPES = {
    'flow_offgas': 0.00012,
    'flow_naphtha': 0.00045,
    'flow_kero': 0.00030,
    'flow_lago': 0.00115,
    'flow_residue': -0.00202
}

def density_to_api(density_val):
    sg = density_val / 1000.0 if density_val > 10.0 else density_val
    if sg <= 0:
        return 30.0
    return (141.5 / sg) - 131.5

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class PhysicsInformedYieldModel:
    def __init__(self):
        self.ml_model = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
        self.linear_reg = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.flow_targets = []
        self.baseline_stats = {}

    def fit(self, X, y_yields, flow_targets, baseline_stats):
        self.flow_targets = flow_targets
        self.baseline_stats = baseline_stats
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y_yields)
        logits = np.log(np.clip(y_yields.values, 1e-4, 1.0))
        self.linear_reg.fit(X_scaled, logits)

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        ml_preds = self.ml_model.predict(X_scaled)
        linear_preds = softmax(self.linear_reg.predict(X_scaled))
        base_yields = 0.70 * ml_preds + 0.30 * linear_preds

        cot = X_df['cot_degC'].values if 'cot_degC' in X_df else self.baseline_stats['mean_cot']
        fzp = X_df['flash_zone_p_kgcm2'].values if 'flash_zone_p_kgcm2' in X_df else self.baseline_stats['mean_p']
        steam = X_df['stripping_steam_flow'].values if 'stripping_steam_flow' in X_df else self.baseline_stats['mean_steam']
        api = X_df['crude_api'].values if 'crude_api' in X_df else self.baseline_stats.get('mean_api', 32.0)

        delta_severity = (
            (cot - self.baseline_stats['mean_cot'])
            - 16.5 * (fzp - self.baseline_stats['mean_p'])
            + 0.55 * (steam - self.baseline_stats['mean_steam'])
            + 0.85 * (api - self.baseline_stats.get('mean_api', 32.0))
        )

        final_yields = np.zeros_like(base_yields)
        for i, col in enumerate(self.flow_targets):
            slope = PHYSICS_SLOPES.get(col, 0.0)
            adj = base_yields[:, i] + (slope * delta_severity)
            min_b, max_b = YIELD_BOUNDS.get(col, (0.01, 0.90))
            final_yields[:, i] = np.clip(adj, min_b, max_b)

        return final_yields / np.sum(final_yields, axis=1, keepdims=True)

# ==============================================================================
# SIDEBAR LOGIN & MULTI-TENANT GATEWAY
# ==============================================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = "Guest"
    st.session_state["role"] = "guest"

st.sidebar.title("🛢️ CDU Digital Twin")

if not st.session_state["authenticated"]:
    with st.sidebar.expander("🔒 Member / Client Login"):
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Sign In"):
            role = verify_login(login_user, login_pass)
            if role:
                st.session_state["authenticated"] = True
                st.session_state["username"] = login_user.strip().lower()
                st.session_state["role"] = role
                log_login_event(login_user)
                st.rerun()
            else:
                st.error("Invalid username or password.")
else:
    st.sidebar.success(f"Logged in as: **{st.session_state['username']}** ({st.session_state['role'].upper()})")
    if st.sidebar.button("Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = "Guest"
        st.session_state["role"] = "guest"
        st.rerun()

nav_options = [
    "1. Model Training & DCS Upload", 
    "2. Yield Prediction",
]

if st.session_state["authenticated"]:
    nav_options.append("3. Protected Workspace & History")

if st.session_state["role"] == "admin":
    nav_options.append("🛡️ Admin Audit & Telemetry")

page = st.sidebar.radio("Navigation", nav_options)

# ==============================================================================
# PAGE 1: GUEST & MEMBER TRAINING INTERFACE
# ==============================================================================
if page == "1. Model Training & DCS Upload":
    st.header("⚙️ Column Data Ingestion & Model Training")
    st.markdown("Upload historical plant logs or click demo to train the hybrid physics-ML twin.")

    uploaded_file = st.file_uploader("Upload DCS Historical Data (CSV or Excel)", type=["csv", "xlsx"])
    col1, col2 = st.columns([1, 4])
    use_synthetic = col1.button("Load Demo DCS Dataset")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        if any("unnamed" in str(col).lower() for col in raw_df.columns):
            raw_df.columns = raw_df.iloc[0].astype(str)
            raw_df = raw_df[1:].reset_index(drop=True)
        st.session_state['active_df'] = raw_df
        st.session_state['active_src_name'] = f"Uploaded File: `{uploaded_file.name}`"
    elif use_synthetic:
        np.random.seed(42)
        n_samples = 1200
        crude_flow = np.random.uniform(350, 450, n_samples)
        cot = np.random.uniform(345, 375, n_samples)
        crude_density = np.random.uniform(840, 890, n_samples)
        api = (141.5 / (crude_density / 1000.0)) - 131.5
        sulfur = np.random.uniform(1.2, 2.8, n_samples)
        fzp = np.random.uniform(1.2, 1.6, n_samples)
        steam = np.random.uniform(8, 14, n_samples)
        lago_t = np.random.uniform(340, 365, n_samples)

        top_reflux = crude_flow * 0.14 + (cot - 360) * 0.4 + np.random.normal(0, 2, n_samples)
        kero_pa = crude_flow * 0.28 + (cot - 360) * 0.8 + np.random.normal(0, 3, n_samples)
        lago_pa = crude_flow * 0.38 + (cot - 360) * 1.2 + np.random.normal(0, 4, n_samples)
        top_t = 120 + 0.15 * (cot - 360) + np.random.normal(0, 1.5, n_samples)
        btm_t = 338 + 0.65 * (cot - 360) - 0.4 * (steam - 10) + np.random.normal(0, 2, n_samples)

        y_offgas = 0.02 + 0.00012 * (cot - 360) + np.random.normal(0, 0.001, n_samples)
        y_naphtha = 0.16 + 0.00045 * (cot - 360) + np.random.normal(0, 0.003, n_samples)
        y_kero = 0.12 + 0.00030 * (cot - 360) + np.random.normal(0, 0.003, n_samples)
        y_lago = 0.28 + 0.00115 * (cot - 360) + np.random.normal(0, 0.004, n_samples)
        y_residue = 1.0 - (y_offgas + y_naphtha + y_kero + y_lago)

        st.session_state['active_df'] = pd.DataFrame({
            'crude_flow': crude_flow, 'crude_density': crude_density, 'crude_api': api,
            'sulfur_wt_pct': sulfur, 'cot_degC': cot, 'flash_zone_p_kgcm2': fzp,
            'stripping_steam_flow': steam, 'lago_d86_95_degC': lago_t,
            'top_reflux_flow_tph': top_reflux, 'kero_pa_flow_tph': kero_pa, 'lago_pa_flow_tph': lago_pa,
            'top_temp_degC': top_t, 'bottom_residue_temp_degC': btm_t,
            'flow_offgas': y_offgas * crude_flow, 'flow_naphtha': y_naphtha * crude_flow,
            'flow_kero': y_kero * crude_flow, 'flow_lago': y_lago * crude_flow,
            'flow_residue': y_residue * crude_flow
        })
        st.session_state['active_src_name'] = "Loaded Synthetic Demo Dataset (1200 records)"

    if 'active_df' in st.session_state:
        df = st.session_state['active_df']
        st.info(f"📂 **Active Dataset:** {st.session_state.get('active_src_name', '')} | Rows: **{len(df)}** | Columns: **{len(df.columns)}**")

        st.subheader("Data Inspector")
        st.dataframe(df.head(5), use_container_width=True)
        with st.expander("🔍 View Complete Raw Dataset"):
            st.dataframe(df)

        has_density = any("dens" in str(c).lower() or "sg" in str(c).lower() for c in df.columns)
        if st.checkbox("Calculate crude_api automatically from Density/SG", value=has_density):
            dens_cols = list(df.columns)
            selected_dens = st.selectbox("Select Density Column", options=dens_cols, index=dens_cols.index('crude_density') if 'crude_density' in dens_cols else 0)
            df['crude_api'] = df[selected_dens].apply(density_to_api)
            st.success(f"Calculated `crude_api` from `{selected_dens}`")

        default_inputs = ['crude_flow', 'crude_api', 'sulfur_wt_pct', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
        default_flows = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
        default_states = ['top_reflux_flow_tph', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC', 'bottom_residue_temp_degC']

        c1, c2, c3 = st.columns(3)
        input_cols = c1.multiselect("Inputs (X)", list(df.columns), default=[c for c in default_inputs if c in df.columns])
        flow_cols = c2.multiselect("Product Flows (Y1)", list(df.columns), default=[c for c in default_flows if c in df.columns])
        state_cols = c3.multiselect("Internal States (Y2)", list(df.columns), default=[c for c in default_states if c in df.columns])
        crude_col = c1.selectbox("Crude Inlet Flow Tag", list(df.columns), index=list(df.columns).index('crude_flow') if 'crude_flow' in df.columns else 0)

        save_as_protected = False
        model_tag = "guest_model"
        if st.session_state["authenticated"]:
            st.divider()
            c_save1, c_save2 = st.columns([1, 2])
            save_as_protected = c_save1.checkbox("Save model into my protected private vault", value=True)
            if save_as_protected:
                model_tag = c_save2.text_input("Protected Model Tag", value=f"{st.session_state['username']}_v1")

        if st.button("🚀 Train Digital Twin", type="primary"):
            with st.spinner("Training model with continuous thermodynamic gradients..."):
                all_needed = list(set(input_cols + flow_cols + state_cols + [crude_col]))
                clean_df = df[all_needed].apply(pd.to_numeric, errors='coerce').dropna()

                total_out = clean_df[flow_cols].sum(axis=1)
                valid_df = clean_df[np.abs(total_out - clean_df[crude_col]) / clean_df[crude_col] < 0.05].copy()

                if valid_df.empty:
                    st.error("❌ Mass balance error: Data does not close within 5%.")
                    st.stop()

                yield_targets = valid_df[flow_cols].div(valid_df[crude_col], axis=0)
                X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te, c_tr, c_te = train_test_split(
                    valid_df[input_cols], yield_targets, valid_df[state_cols], valid_df[crude_col], test_size=0.2, random_state=42
                )

                baseline_stats = {
                    'mean_cot': float(valid_df['cot_degC'].mean()) if 'cot_degC' in valid_df.columns else 350.0,
                    'mean_p': float(valid_df['flash_zone_p_kgcm2'].mean()) if 'flash_zone_p_kgcm2' in valid_df.columns else 1.40,
                    'mean_steam': float(valid_df['stripping_steam_flow'].mean()) if 'stripping_steam_flow' in valid_df.columns else 10.0,
                    'mean_api': float(valid_df['crude_api'].mean()) if 'crude_api' in valid_df.columns else 32.0
                }

                yield_model = PhysicsInformedYieldModel()
                yield_model.fit(X_tr, yf_tr, flow_cols, baseline_stats)

                scaler_states = StandardScaler()
                model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
                model_states.fit(scaler_states.fit_transform(X_tr), ys_tr)

                pipeline = {
                    "yield_model": yield_model,
                    "model_states": model_states,
                    "scaler_states": scaler_states,
                    "input_cols": input_cols,
                    "flow_targets": flow_cols,
                    "state_targets": state_cols,
                    "crude_col": crude_col,
                    "baseline_stats": baseline_stats,
                    "training_rows": len(valid_df),
                    "last_known_inputs": clean_df[input_cols].iloc[-1].to_dict()
                }

                if save_as_protected and st.session_state["authenticated"]:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"models/protected_{st.session_state['username']}_{timestamp_str}.pkl"
                    joblib.dump(pipeline, save_path)
                    
                    conn = get_db_connection()
                    conn.execute('''
                        INSERT INTO protected_models (username, model_tag, model_path, training_rows, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (st.session_state["username"], model_tag, save_path, len(valid_df), datetime.now()))
                    conn.commit()
                    conn.close()
                    st.success(f"🔒 Model saved to your private vault as `{model_tag}`!")
                else:
                    joblib.dump(pipeline, GUEST_MODEL_FILE)
                    st.success("🌐 Model trained and saved into public guest sandbox.")

# ==============================================================================
# PAGE 2: REAL-TIME PREDICTION & CONTINUOUS SENSITIVITY
# ==============================================================================
elif page == "2. Yield Prediction":
    st.header("🎯 Autonomous CDU Prediction & Dynamic Sensitivity")

    active_pipeline = None
    if st.session_state["authenticated"]:
        conn = get_db_connection()
        user_models = conn.execute("SELECT id, model_tag, model_path FROM protected_models WHERE username = ?", 
                                   (st.session_state["username"],)).fetchall()
        conn.close()

        source_choice = st.radio("Prediction Model Source:", ["Public Guest Sandbox Model", "My Private Vault Models"], horizontal=True)
        if source_choice == "My Private Vault Models" and user_models:
            chosen = st.selectbox("Select Private Model", options=user_models, format_func=lambda x: x["model_tag"])
            if chosen and os.path.exists(chosen["model_path"]):
                active_pipeline = joblib.load(chosen["model_path"])
        else:
            if os.path.exists(GUEST_MODEL_FILE):
                active_pipeline = joblib.load(GUEST_MODEL_FILE)
    else:
        if os.path.exists(GUEST_MODEL_FILE):
            active_pipeline = joblib.load(GUEST_MODEL_FILE)

    if not active_pipeline:
        st.warning("⚠️ No trained model found. Please train a model on Page 1 first.")
    else:
        input_cols = active_pipeline["input_cols"]
        flow_targets = active_pipeline["flow_targets"]
        state_targets = active_pipeline["state_targets"]
        last_in = active_pipeline.get("last_known_inputs", {})
        stats = active_pipeline["baseline_stats"]

        st.subheader("1. Crude Assay Properties")
        c_dens1, c_dens2 = st.columns(2)
        last_api = float(last_in.get('crude_api', 32.0))
        default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 865.0

        input_density = c_dens1.number_input("Crude Density (kg/m³ or SG @ 15°C)", value=default_density, format="%.2f")
        calculated_api = density_to_api(input_density)
        c_dens2.metric("Calculated Crude API", f"{calculated_api:.2f} °API")

        st.subheader("2. Operating Boundary Inputs (Last Input Initialized)")
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(input_cols):
            if feat == 'crude_api':
                input_data[feat] = calculated_api
            else:
                fallback = last_in.get(feat, 360.0 if "cot" in feat.lower() else (400.0 if "crude" in feat.lower() else 1.4))
                input_data[feat] = cols[i % 3].number_input(feat, value=float(fallback), format="%.2f")

        if st.button("🔮 Run Simulation & Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            norm_yields = active_pipeline["yield_model"].predict(input_df)[0]
            crude_in = input_data[active_pipeline["crude_col"]]
            pred_flows = norm_yields * crude_in

            scaled_state = active_pipeline["scaler_states"].transform(input_df)
            pred_states = active_pipeline["model_states"].predict(scaled_state)[0]

            # Dynamic thermal adjustments
            cot_delta = input_data.get('cot_degC', 350.0) - stats['mean_cot']
            fzp_delta = input_data.get('flash_zone_p_kgcm2', 1.40) - stats['mean_p']

            for k, s in enumerate(state_targets):
                if "bottom_residue_temp" in s.lower():
                    pred_states[k] += 0.65 * cot_delta - 8.0 * fzp_delta
                elif "top_temp" in s.lower():
                    pred_states[k] += 0.15 * cot_delta - 2.5 * fzp_delta
                elif "pa" in s.lower():
                    pred_states[k] += 0.80 * cot_delta

            if st.session_state["authenticated"]:
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO simulation_history (username, timestamp, inputs_json, outputs_json)
                    VALUES (?, ?, ?, ?)
                ''', (st.session_state["username"], datetime.now(), json.dumps(input_data), json.dumps(dict(zip(flow_targets, pred_flows)))))
                conn.commit()
                conn.close()

            st.divider()
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("📦 Product Recovery Yields")
                st.table(pd.DataFrame({
                    "Cut Stream": flow_targets,
                    "Yield (wt%)": [f"{y*100:.2f}%" for y in norm_yields],
                    "Rate (t/h)": [f"{f:.2f}" for f in pred_flows]
                }))
                st.metric("Total Mass Out", f"{np.sum(pred_flows):.2f} t/h (Closure: 0.00% error)")

            with c_right:
                st.subheader("🌡️ Predicted Column Profile")
                st.table(pd.DataFrame({
                    "Parameter": state_targets,
                    "Predicted Value": [f"{v:.2f} {'°C' if 'temp' in n.lower() else 't/h'}" for n, v in zip(state_targets, pred_states)]
                }))

# ==============================================================================
# PAGE 3: PROTECTED WORKSPACE (MEMBER EXCLUSIVE)
# ==============================================================================
elif page == "3. Protected Workspace & History":
    st.header(f"🔒 Protected Workspace: `{st.session_state['username']}`")
    conn = get_db_connection()

    tab_my_models, tab_my_sims = st.tabs(["📁 My Saved Models", "📜 My Simulation History"])

    with tab_my_models:
        st.subheader("Your Isolated Models")
        my_models = pd.read_sql_query(
            "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ? ORDER BY created_at DESC",
            conn, params=(st.session_state['username'],)
        )
        if my_models.empty:
            st.info("No protected models saved yet. Train one on Page 1 while signed in.")
        else:
            st.dataframe(my_models, use_container_width=True)

    with tab_my_sims:
        st.subheader("Your Saved Simulations")
        my_sims = pd.read_sql_query(
            "SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC",
            conn, params=(st.session_state['username'],)
        )
        if my_sims.empty:
            st.info("No logged simulations on record.")
        else:
            st.dataframe(my_sims, use_container_width=True)

    conn.close()

# ==============================================================================
# PAGE 4: ADMIN GOVERNANCE & TELEMETRY (ADMIN EXCLUSIVE)
# ==============================================================================
elif page == "🛡️ Admin Audit & Telemetry":
    st.header("🛡️ Enterprise Client Governance & Audit Portal")
    conn = get_db_connection()

    tab_manage, tab_client_inspect, tab_logs = st.tabs([
        "👥 Client Account Provisioning", 
        "🔍 Inspect & Modify Client Spaces", 
        "📍 Access & Location Audit"
    ])

    with tab_manage:
        st.subheader("Create New Client Account")
        c_u1, c_u2, c_u3 = st.columns([2, 2, 1])
        new_client_user = c_u1.text_input("New Client Username", placeholder="e.g. refinery_client_a")
        new_client_pass = c_u2.text_input("Initial Password", placeholder="e.g. Pass@2026")
        
        if c_u3.button("Create Account", type="primary"):
            if new_client_user and new_client_pass:
                ok, msg = create_client_user(new_client_user, new_client_pass)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please provide both username and password.")

        st.divider()
        st.subheader("Existing Accounts")
        users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY role ASC, username ASC", conn)
        st.dataframe(users_df, use_container_width=True)

        st.markdown("#### Password Reset / Account Management")
        client_list = [u for u in users_df["username"].tolist() if u != "admin"]
        if client_list:
            c_sel, c_np, c_btn1, c_btn2 = st.columns([2, 2, 1, 1])
            selected_client = c_sel.selectbox("Select Client", options=client_list)
            reset_pw = c_np.text_input("New Password", placeholder="Enter new password")
            
            if c_btn1.button("Reset Password"):
                if reset_pw:
                    reset_client_password(selected_client, reset_pw)
                    st.success(f"Password updated for `{selected_client}`.")
                else:
                    st.error("Enter a valid password.")

            if c_btn2.button("Delete Client", type="secondary"):
                delete_client_user(selected_client)
                st.warning(f"Client `{selected_client}` deleted.")
                st.rerun()

    with tab_client_inspect:
        st.subheader("Client Data Inspector & Editor")
        all_clients = [u for u in users_df["username"].tolist() if u != "admin"]

        if not all_clients:
            st.info("No registered clients available to inspect.")
        else:
            chosen_user = st.selectbox("Select Client Profile to Inspect:", options=all_clients)
            
            col_m, col_s = st.columns(2)
            with col_m:
                st.markdown(f"**Models Saved by `{chosen_user}`:**")
                client_models = pd.read_sql_query(
                    "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ?",
                    conn, params=(chosen_user,)
                )
                if client_models.empty:
                    st.caption("No models saved by this client.")
                else:
                    st.dataframe(client_models, use_container_width=True)
                    del_m_id = st.selectbox("Delete Model ID", options=client_models["id"].tolist(), key="del_m_key")
                    if st.button("Delete Selected Model", key="del_m_btn"):
                        m_row = client_models[client_models["id"] == del_m_id].iloc[0]
                        if os.path.exists(m_row["model_path"]):
                            os.remove(m_row["model_path"])
                        conn.execute("DELETE FROM protected_models WHERE id = ?", (del_m_id,))
                        conn.commit()
                        st.success(f"Removed model `{m_row['model_tag']}`.")
                        st.rerun()

            with col_s:
                st.markdown(f"**Simulations Run by `{chosen_user}`:**")
                client_sims = pd.read_sql_query(
                    "SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC",
                    conn, params=(chosen_user,)
                )
                if client_sims.empty:
                    st.caption("No simulations logged for this client.")
                else:
                    st.dataframe(client_sims, use_container_width=True)

    with tab_logs:
        st.subheader("Global Sign-in Geolocation & Telemetry")
        access_df = pd.read_sql_query("SELECT username, login_time, ip_address, city, region, country FROM access_logs ORDER BY login_time DESC", conn)
        st.dataframe(access_df, use_container_width=True)

    conn.close()
