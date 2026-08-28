import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- App Config ---
st.set_page_config(page_title="CDU Autonomous Digital Twin", layout="wide")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_FILE = "models/cdu_full_twin_pipeline.pkl"
DATA_FILE = "data/master_cdu_dataset.parquet"

# Physical Engineering Bounds for Atmospheric Cuts (Prevents overestimating Off-Gas / LPG)
YIELD_BOUNDS = {
    'flow_offgas': (0.005, 0.040),    # 0.5% to 4.0% of crude
    'flow_naphtha': (0.080, 0.300),   # 8% to 30%
    'flow_kero': (0.050, 0.220),      # 5% to 22%
    'flow_lago': (0.150, 0.420),      # 15% to 42%
    'flow_residue': (0.250, 0.650)    # 25% to 65%
}

def density_to_api(density_val):
    sg = density_val / 1000.0 if density_val > 10.0 else density_val
    if sg <= 0:
        return 30.0
    return (141.5 / sg) - 131.5

DEFAULT_INPUTS = [
    'crude_flow', 'crude_api', 'sulfur_wt_pct',
    'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow',
    'lago_d86_95_degC'
]

DEFAULT_PRODUCT_TARGETS = [
    'flow_offgas', 'flow_naphtha', 'flow_kero', 
    'flow_lago', 'flow_residue'
]

DEFAULT_STATE_TARGETS = [
    'top_reflux_flow_tph', 'kero_pa_flow_tph', 'lago_pa_flow_tph',
    'top_temp_degC', 'bottom_residue_temp_degC'
]

def train_and_save_pipeline(train_df, input_cols, flow_target_cols, state_target_cols, crude_flow_col):
    all_needed_cols = list(set(input_cols + flow_target_cols + state_target_cols + [crude_flow_col]))
    clean_df = train_df[all_needed_cols].copy()
    
    for col in all_needed_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    clean_df = clean_df.dropna()

    if clean_df.empty:
        return False, "No valid numerical data found.", None

    # Mass balance filter (<3% error)
    total_out = clean_df[flow_target_cols].sum(axis=1)
    imbalance = np.abs(total_out - clean_df[crude_flow_col]) / clean_df[crude_flow_col]
    valid_df = clean_df[imbalance < 0.03].copy()

    if valid_df.empty:
        return False, "Mass balance error: No rows had mass closure within 3%. Check units across flows.", None

    yield_targets = valid_df[flow_target_cols].div(valid_df[crude_flow_col], axis=0)
    
    X = valid_df[input_cols]
    y_flows = yield_targets
    y_states = valid_df[state_target_cols]

    X_train, X_test, yf_train, yf_test, ys_train, ys_test, crude_train, crude_test = train_test_split(
        X, y_flows, y_states, valid_df[crude_flow_col], test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Product Cuts
    model_flows = MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42))
    model_flows.fit(X_train_scaled, yf_train)

    # Model 2: Pumparounds & Temperatures
    model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42))
    model_states.fit(X_train_scaled, ys_train)

    # 1. Physical Bounding & Normalization on Test Predictions
    raw_yield_preds = model_flows.predict(X_test_scaled)
    bounded_yield_preds = np.zeros_like(raw_yield_preds)
    for i, col in enumerate(flow_target_cols):
        min_val, max_val = YIELD_BOUNDS.get(col, (0.0, 1.0))
        bounded_yield_preds[:, i] = np.clip(raw_yield_preds[:, i], min_val, max_val)

    norm_yield_preds = bounded_yield_preds / bounded_yield_preds.sum(axis=1, keepdims=True)
    pred_flows = norm_yield_preds * crude_test.values.reshape(-1, 1)
    actual_flows = (yf_test.values * crude_test.values.reshape(-1, 1))

    pred_states = model_states.predict(X_test_scaled)
    actual_states = ys_test.values

    metrics = []
    for i, col in enumerate(flow_target_cols):
        r2 = r2_score(actual_flows[:, i], pred_flows[:, i])
        mae = mean_absolute_error(actual_flows[:, i], pred_flows[:, i])
        metrics.append({"Target": col, "Category": "Product Stream", "R² Score": round(float(r2), 4), "MAE": f"{mae:.2f} t/h"})

    for i, col in enumerate(state_target_cols):
        r2 = r2_score(actual_states[:, i], pred_states[:, i])
        mae = mean_absolute_error(actual_states[:, i], pred_states[:, i])
        unit = "°C" if "temp" in col.lower() else "t/h"
        metrics.append({"Target": col, "Category": "Internal State / PA", "R² Score": round(float(r2), 4), "MAE": f"{mae:.2f} {unit}"})

    last_known_inputs = clean_df[input_cols].iloc[-1].to_dict()

    pipeline = {
        "model_flows": model_flows,
        "model_states": model_states,
        "scaler": scaler,
        "input_cols": input_cols,
        "flow_targets": flow_target_cols,
        "state_targets": state_target_cols,
        "crude_col": crude_flow_col,
        "metrics": metrics,
        "training_rows": len(valid_df),
        "last_known_inputs": last_known_inputs
    }
    joblib.dump(pipeline, MODEL_FILE)
    clean_df.to_parquet(DATA_FILE, index=False)
    return True, "Success", metrics

# --- Sidebar Navigation ---
st.sidebar.title("🛢️ CDU Digital Twin")
page = st.sidebar.radio("Navigation", ["1. Model Training & Upload", "2. Minimal Input Prediction", "3. Model Management & Data Appending"])

# ==============================================================================
# PAGE 1: TRAINING INTERFACE
# ==============================================================================
if page == "1. Model Training & Upload":
    st.header("⚙️ CDU Data Ingestion & Digital Twin Training")
    st.markdown("Upload your column historical Excel/CSV file or load the demo dataset to train the dual regression model.")

    uploaded_file = st.file_uploader("Upload DCS Historical Data (CSV or Excel)", type=["csv", "xlsx"])
    col1, col2 = st.columns([1, 4])
    use_synthetic = col1.button("Load Demo DCS Dataset")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        if any("unnamed" in str(col).lower() for col in raw_df.columns):
            raw_df.columns = raw_df.iloc[0].astype(str)
            raw_df = raw_df[1:].reset_index(drop=True)
        st.session_state['active_df'] = raw_df
        st.session_state['data_source_name'] = f"Uploaded File: `{uploaded_file.name}`"
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

        top_reflux = crude_flow * 0.14 + np.random.normal(0, 2, n_samples)
        kero_pa = crude_flow * 0.28 + (cot - 360) * 0.8 + np.random.normal(0, 3, n_samples)
        lago_pa = crude_flow * 0.38 + (cot - 360) * 1.2 + np.random.normal(0, 4, n_samples)
        top_t = 120 + 0.15 * (cot - 360) + np.random.normal(0, 1.5, n_samples)
        btm_t = 338 + 0.65 * (cot - 360) - 0.4 * (steam - 10) + np.random.normal(0, 2, n_samples)

        y_offgas = 0.02 + 0.0003 * (cot - 360) + np.random.normal(0, 0.002, n_samples)
        y_naphtha = 0.16 + 0.0005 * (cot - 360) + np.random.normal(0, 0.005, n_samples)
        y_kero = 0.12 + 0.0002 * (cot - 360) + np.random.normal(0, 0.004, n_samples)
        y_lago = 0.28 + 0.0012 * (cot - 360) + np.random.normal(0, 0.006, n_samples)
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
        st.session_state['data_source_name'] = "Loaded Synthetic Demo Dataset (1200 records)"

    if 'active_df' in st.session_state:
        df = st.session_state['active_df']
        st.info(f"📂 **Active Source:** {st.session_state.get('data_source_name', 'Active Dataset')} | Total Rows: **{len(df)}** | Columns: **{len(df.columns)}**")

        st.subheader("Data Inspector")
        st.dataframe(df.head(10), use_container_width=True)
        with st.expander("🔍 View Complete Raw Dataset"):
            st.dataframe(df)

        st.subheader("Crude Density / API Configuration")
        has_density = any("dens" in str(c).lower() or "sg" in str(c).lower() for c in df.columns)
        if st.checkbox("Calculate crude_api automatically from Density/SG column", value=has_density):
            dens_cols = list(df.columns)
            selected_dens = st.selectbox("Select Crude Density / SG Column", options=dens_cols, index=dens_cols.index('crude_density') if 'crude_density' in dens_cols else 0)
            df['crude_api'] = df[selected_dens].apply(density_to_api)
            st.success(f"Calculated `crude_api` from `{selected_dens}`")

        st.subheader("Column Mapping")
        c1, c2, c3 = st.columns(3)
        input_cols = c1.multiselect("Minimal Boundary Inputs (X)", options=list(df.columns), default=[c for c in DEFAULT_INPUTS if c in df.columns])
        flow_target_cols = c2.multiselect("Product Cut Flows (Y1)", options=list(df.columns), default=[c for c in DEFAULT_PRODUCT_TARGETS if c in df.columns])
        state_target_cols = c3.multiselect("Internal States / PA / Temps (Y2)", options=list(df.columns), default=[c for c in DEFAULT_STATE_TARGETS if c in df.columns])
        crude_flow_col = c1.selectbox("Crude Inlet Flow Tag", options=list(df.columns), index=list(df.columns).index('crude_flow') if 'crude_flow' in df.columns else 0)

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training predictive models for flows, pumparounds, and temperatures..."):
                success, msg, metrics = train_and_save_pipeline(df, input_cols, flow_target_cols, state_target_cols, crude_flow_col)
                if success:
                    st.success("Digital twin trained and saved successfully!")
                    st.subheader("📊 Performance Metrics on Validation Set")
                    st.table(pd.DataFrame(metrics))
                else:
                    st.error(f"❌ {msg}")

# ==============================================================================
# PAGE 2: PREDICTION INTERFACE (MINIMAL INPUTS ONLY)
# ==============================================================================
elif page == "2. Minimal Input Prediction":
    st.header("🎯 Autonomous CDU Prediction")
    st.markdown("Enter boundary conditions (Feed, COT, Pressure) to predict product cuts, pumparounds, and temperatures.")

    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ No trained model found. Please train a model on Page 1 first.")
    else:
        pipeline = joblib.load(MODEL_FILE)
        input_cols = pipeline["input_cols"]
        flow_targets = pipeline["flow_targets"]
        state_targets = pipeline["state_targets"]
        last_inputs = pipeline.get("last_known_inputs", {})

        st.subheader("1. Crude Assay Properties")
        c_dens1, c_dens2 = st.columns(2)
        
        last_api = float(last_inputs.get('crude_api', 32.0))
        default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 865.0

        input_density = c_dens1.number_input("Crude Density (kg/m³ or SG @ 15°C)", value=default_density, format="%.2f")
        calculated_api = density_to_api(input_density)
        c_dens2.metric("Calculated Crude API", f"{calculated_api:.2f} °API")

        st.subheader("2. Operating Boundary Inputs (Initialized with Last Inputted Snapshot)")
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(input_cols):
            col = cols[i % 3]
            if feat == 'crude_api':
                input_data[feat] = calculated_api
            else:
                fallback_val = last_inputs.get(feat, 360.0 if "cot" in feat.lower() else (400.0 if "crude_flow" in feat.lower() else (1.4 if "p_kgcm2" in feat.lower() else 355.0)))
                input_data[feat] = col.number_input(f"{feat}", value=float(fallback_val), format="%.2f")

        if st.button("🔮 Run Simulation & Predict", type="primary"):
            pipeline["last_known_inputs"] = input_data
            joblib.dump(pipeline, MODEL_FILE)

            input_df = pd.DataFrame([input_data])
            scaled_input = pipeline["scaler"].transform(input_df)

            # 1. Raw Prediction & Hard Physical Bounding
            raw_yields = pipeline["model_flows"].predict(scaled_input)[0]
            bounded_yields = np.zeros_like(raw_yields)
            for i, col in enumerate(flow_targets):
                min_val, max_val = YIELD_BOUNDS.get(col, (0.0, 1.0))
                bounded_yields[i] = np.clip(raw_yields[i], min_val, max_val)

            # 2. Strict Mass Balance Closure
            norm_yields = bounded_yields / np.sum(bounded_yields)

            crude_in = input_data[pipeline["crude_col"]]
            pred_flows = norm_yields * crude_in

            # 3. Predict Internal Column States
            pred_states = pipeline["model_states"].predict(scaled_input)[0]

            st.divider()
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("📦 Predicted Product Cuts & Yields")
                res_df = pd.DataFrame({
                    "Product Stream": flow_targets,
                    "Yield (wt%)": [f"{y*100:.2f}%" for y in norm_yields],
                    "Flow Rate (t/h)": [f"{f:.2f}" for f in pred_flows]
                })
                st.table(res_df)
                st.metric("Total Crude In", f"{crude_in:.2f} t/h")
                st.metric("Total Mass Out", f"{np.sum(pred_flows):.2f} t/h (Closure: 0.00% error)")

            with col_right:
                st.subheader("🌡️ Predicted Internal Column Profile")
                state_data = []
                for name, val in zip(state_targets, pred_states):
                    unit = "°C" if "temp" in name.lower() else "t/h"
                    state_data.append({"Parameter": name, "Predicted Value": f"{val:.2f} {unit}"})
                st.table(pd.DataFrame(state_data))

# ==============================================================================
# PAGE 3: MODEL MANAGEMENT & DATA APPENDING
# ==============================================================================
elif page == "3. Model Management & Data Appending":
    st.header("📊 Model Overview & Continuous Learning")

    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ No trained model found. Please train a model on Page 1 first.")
    else:
        pipeline = joblib.load(MODEL_FILE)
        st.subheader("🔍 Active Digital Twin Architecture")
        m1, m2, m3 = st.columns(3)
        m1.metric("Architecture", "Dual LightGBM Engine (Physically Bounded)")
        m2.metric("Trained Samples", f"{pipeline.get('training_rows', 'N/A')} snapshots")
        m3.metric("Boundary Inputs", f"{len(pipeline['input_cols'])} variables")

        if pipeline.get("metrics"):
            st.markdown("**Test Set Accuracy Breakdown:**")
            st.table(pd.DataFrame(pipeline["metrics"]))

        if os.path.exists(DATA_FILE):
            st.divider()
            st.subheader("💾 Export Current Master Training Dataset")
            export_df = pd.read_parquet(DATA_FILE)
            st.download_button("📥 Download Master Dataset (CSV)", export_df.to_csv(index=False).encode('utf-8'), "cdu_twin_data.csv", "text/csv")
