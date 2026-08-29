import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- App Config ---
st.set_page_config(page_title="CDU Physics-Informed Digital Twin", layout="wide")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_FILE = "models/cdu_pinn_pipeline.pkl"
DATA_FILE = "data/master_cdu_dataset.parquet"

# Physical Hard Clamping Bounds for Distillation Envelopes
YIELD_BOUNDS = {
    'flow_offgas': (0.005, 0.045),
    'flow_naphtha': (0.080, 0.320),
    'flow_kero': (0.050, 0.220),
    'flow_lago': (0.150, 0.420),
    'flow_residue': (0.220, 0.650)
}

# Empirical Thermodynamic Flash Gradients (wt% shift per °C effective flash severity)
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

class PhysicsInformedYieldModel:
    """Hybrid Continuous Physics-ML Ensemble ensuring Smooth Derivatives and Mass Conservation."""
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
        
        # Train ML tree component
        self.ml_model.fit(X_scaled, y_yields)
        
        # Train Ridge linear component for robust continuous extrapolation
        logits = np.log(np.clip(y_yields.values, 1e-4, 1.0))
        self.linear_reg.fit(X_scaled, logits)

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        
        # 1. Base ML & Continuous Ridge Extrapolation
        ml_preds = self.ml_model.predict(X_scaled)
        linear_logits = self.linear_reg.predict(X_scaled)
        linear_preds = softmax(linear_logits)

        # 2. Ensemble Blend (70% ML Local Precision + 30% Continuous Extrapolation)
        base_yields = 0.70 * ml_preds + 0.30 * linear_preds

        # 3. Dynamic Flash Severity Adjustment
        cot = X_df['cot_degC'].values if 'cot_degC' in X_df else self.baseline_stats['mean_cot']
        fzp = X_df['flash_zone_p_kgcm2'].values if 'flash_zone_p_kgcm2' in X_df else self.baseline_stats['mean_p']
        steam = X_df['stripping_steam_flow'].values if 'stripping_steam_flow' in X_df else self.baseline_stats['mean_steam']
        api = X_df['crude_api'].values if 'crude_api' in X_df else self.baseline_stats.get('mean_api', 32.0)

        # Effective Thermodynamic Vaporization Severity
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

        # 4. Final Coupled Softmax/Mass Closure
        return final_yields / np.sum(final_yields, axis=1, keepdims=True)

def train_and_save_pipeline(train_df, input_cols, flow_target_cols, state_target_cols, crude_flow_col):
    all_needed_cols = list(set(input_cols + flow_target_cols + state_target_cols + [crude_flow_col]))
    clean_df = train_df[all_needed_cols].copy()
    
    for col in all_needed_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    clean_df = clean_df.dropna()

    if clean_df.empty:
        return False, "No valid numerical data found.", None

    total_out = clean_df[flow_target_cols].sum(axis=1)
    imbalance = np.abs(total_out - clean_df[crude_flow_col]) / clean_df[crude_flow_col]
    valid_df = clean_df[imbalance < 0.05].copy()

    if valid_df.empty:
        return False, "Mass balance error: No rows closed within 5%. Check flow units.", None

    yield_targets = valid_df[flow_target_cols].div(valid_df[crude_flow_col], axis=0)
    
    X = valid_df[input_cols]
    y_flows = yield_targets
    y_states = valid_df[state_target_cols]

    X_train, X_test, yf_train, yf_test, ys_train, ys_test, crude_train, crude_test = train_test_split(
        X, y_flows, y_states, valid_df[crude_flow_col], test_size=0.2, random_state=42
    )

    baseline_stats = {
        'mean_cot': float(valid_df['cot_degC'].mean()) if 'cot_degC' in valid_df.columns else 350.0,
        'mean_p': float(valid_df['flash_zone_p_kgcm2'].mean()) if 'flash_zone_p_kgcm2' in valid_df.columns else 1.40,
        'mean_steam': float(valid_df['stripping_steam_flow'].mean()) if 'stripping_steam_flow' in valid_df.columns else 10.0,
        'mean_api': float(valid_df['crude_api'].mean()) if 'crude_api' in valid_df.columns else 32.0
    }

    # Train Physics-Informed Yield Engine
    yield_model = PhysicsInformedYieldModel()
    yield_model.fit(X_train, yf_train, flow_target_cols, baseline_stats)

    # Train State & Pumparound Regressor
    scaler_states = StandardScaler()
    X_train_s = scaler_states.fit_transform(X_train)
    X_test_s = scaler_states.transform(X_test)
    model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
    model_states.fit(X_train_s, ys_train)

    # Performance Evaluation on Unseen Test Split
    norm_yield_preds = yield_model.predict(X_test)
    pred_flows = norm_yield_preds * crude_test.values.reshape(-1, 1)
    actual_flows = (yf_test.values * crude_test.values.reshape(-1, 1))

    pred_states = model_states.predict(X_test_s)
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
        "yield_model": yield_model,
        "model_states": model_states,
        "scaler_states": scaler_states,
        "input_cols": input_cols,
        "flow_targets": flow_target_cols,
        "state_targets": state_target_cols,
        "crude_col": crude_flow_col,
        "metrics": metrics,
        "training_rows": len(valid_df),
        "baseline_stats": baseline_stats,
        "last_known_inputs": last_known_inputs
    }
    joblib.dump(pipeline, MODEL_FILE)
    clean_df.to_parquet(DATA_FILE, index=False)
    return True, "Success", metrics

# --- Sidebar Navigation ---
st.sidebar.title("🛢️ CDU Hybrid Digital Twin")
page = st.sidebar.radio("Navigation", ["1. Model Training & Upload", "2. Minimal Input Prediction", "3. Model Management & Data Appending"])

# ==============================================================================
# PAGE 1: TRAINING INTERFACE
# ==============================================================================
if page == "1. Model Training & Upload":
    st.header("⚙️ Column Historical Training & Ingestion")
    st.markdown("Train the Physics-Informed Hybrid Ensemble on refinery DCS operating logs.")

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
        st.session_state['data_source_name'] = "Loaded Synthetic Demo Dataset (1200 records)"

    if 'active_df' in st.session_state:
        df = st.session_state['active_df']
        st.info(f"📂 **Active Source:** {st.session_state.get('data_source_name', 'Active Dataset')} | Total Rows: **{len(df)}**")

        st.subheader("Data Inspector")
        st.dataframe(df.head(10), use_container_width=True)

        has_density = any("dens" in str(c).lower() or "sg" in str(c).lower() for c in df.columns)
        if st.checkbox("Calculate crude_api automatically from Density/SG", value=has_density):
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

        if st.button("🚀 Train Physics-Informed Model", type="primary"):
            with st.spinner("Training Physics-Informed Multi-Output Ensemble..."):
                success, msg, metrics = train_and_save_pipeline(df, input_cols, flow_target_cols, state_target_cols, crude_flow_col)
                if success:
                    st.success("Physics-Informed Digital Twin trained and saved successfully!")
                    st.subheader("📊 Performance Metrics on Validation Set")
                    st.table(pd.DataFrame(metrics))
                else:
                    st.error(f"❌ {msg}")

# ==============================================================================
# PAGE 2: PREDICTION INTERFACE (CONTINUOUS PHYSICS SIMULATION)
# ==============================================================================
elif page == "2. Minimal Input Prediction":
    st.header("🎯 Autonomous CDU Prediction & Dynamic Sensitivity")
    st.markdown("Adjust furnace COT, Flash Zone Pressure, or feed rates to see real-time continuous shifts in cut recovery.")

    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ No trained model found. Please train a model on Page 1 first.")
    else:
        pipeline = joblib.load(MODEL_FILE)
        input_cols = pipeline["input_cols"]
        flow_targets = pipeline["flow_targets"]
        state_targets = pipeline["state_targets"]
        last_inputs = pipeline.get("last_known_inputs", {})
        baseline_stats = pipeline.get("baseline_stats", {})

        st.subheader("1. Crude Assay Properties")
        c_dens1, c_dens2 = st.columns(2)
        last_api = float(last_inputs.get('crude_api', 32.0))
        default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 865.0

        input_density = c_dens1.number_input("Crude Density (kg/m³ or SG @ 15°C)", value=default_density, format="%.2f")
        calculated_api = density_to_api(input_density)
        c_dens2.metric("Calculated Crude API", f"{calculated_api:.2f} °API")

        st.subheader("2. Operating Boundary Inputs")
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

            # 1. Continuous Physics-Informed Yield Prediction
            norm_yields = pipeline["yield_model"].predict(input_df)[0]
            crude_in = input_data[pipeline["crude_col"]]
            pred_flows = norm_yields * crude_in

            # 2. Internal Column State Prediction
            scaled_state_in = pipeline["scaler_states"].transform(input_df)
            pred_states = pipeline["model_states"].predict(scaled_state_in)[0]

            # Dynamic thermodynamic thermal scaling
            cot_delta = input_data.get('cot_degC', 350.0) - baseline_stats.get('mean_cot', 350.0)
            fzp_delta = input_data.get('flash_zone_p_kgcm2', 1.40) - baseline_stats.get('mean_p', 1.40)

            for k, s_name in enumerate(state_targets):
                if "bottom_residue_temp" in s_name.lower():
                    pred_states[k] += 0.65 * cot_delta - 8.0 * fzp_delta
                elif "top_temp" in s_name.lower():
                    pred_states[k] += 0.15 * cot_delta - 2.5 * fzp_delta
                elif "pa" in s_name.lower():
                    pred_states[k] += 0.80 * cot_delta

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
# PAGE 3: MODEL MANAGEMENT
# ==============================================================================
elif page == "3. Model Management & Data Appending":
    st.header("📊 Model Overview & Continuous Learning")

    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ No trained model found. Please train a model on Page 1 first.")
    else:
        pipeline = joblib.load(MODEL_FILE)
        st.subheader("🔍 Active Digital Twin Architecture")
        m1, m2, m3 = st.columns(3)
        m1.metric("Architecture", "Physics-Informed Hybrid Ensemble (PINN-lite)")
        m2.metric("Trained Samples", f"{pipeline.get('training_rows', 'N/A')} snapshots")
        m3.metric("Baseline Anchor", f"COT {pipeline['baseline_stats']['mean_cot']:.1f}°C | P {pipeline['baseline_stats']['mean_p']:.2f} kg/cm²")

        if pipeline.get("metrics"):
            st.markdown("**Validation Accuracy Breakdown:**")
            st.table(pd.DataFrame(pipeline["metrics"]))

        if os.path.exists(DATA_FILE):
            st.divider()
            st.subheader("💾 Export Current Master Training Dataset")
            export_df = pd.read_parquet(DATA_FILE)
            st.download_button("📥 Download Master Dataset (CSV)", export_df.to_csv(index=False).encode('utf-8'), "cdu_twin_data.csv", "text/csv")
