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
st.title('cdu predictor')
st.write('crude distilation column predictor')

# --- App Config ---
st.set_page_config(page_title="CDU Cut & Yield Predictor", layout="wide")
os.makedirs("models", exist_ok=True)
MODEL_FILE = "models/cdu_model_pipeline.pkl"

DEFAULT_FEATURES = [
    'crude_flow', 'crude_api', 'sulfur_wt_pct',
    'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow',
    'reflux_ratio', 'top_temp_degC', 'lago_d86_95_degC'
]

DEFAULT_TARGETS = [
    'flow_offgas', 'flow_naphtha', 'flow_kero', 
    'flow_lago', 'flow_residue'
]

# --- Sidebar Navigation ---
st.sidebar.title("🛢️ CDU ML Platform")
page = st.sidebar.radio("Navigation", ["1. Model Training & DCS Upload", "2. Real-Time Yield Prediction"])

# ==============================================================================
# PAGE 1: TRAINING INTERFACE
# ==============================================================================
if page == "1. Model Training & DCS Upload":
    st.header("⚙️ Column Data Ingestion & Model Training")
    st.markdown("Upload your CDU DCS dataset (CSV/Excel) with operating parameters and outlet flows.")

    # Generate or upload dataset
    uploaded_file = st.file_uploader("Upload DCS Historical Data", type=["csv", "xlsx"])
    
    col1, col2 = st.columns([1, 4])
    use_synthetic = col1.button("Load Demo DCS Dataset")

    if uploaded_file is not None or use_synthetic:
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Generate synthetic plant dataset
            np.random.seed(42)
            n_samples = 1200
            crude_flow = np.random.uniform(350, 450, n_samples)
            cot = np.random.uniform(345, 375, n_samples)
            oil_density = np.random.uniform(873.2, 875.3, n_samples)
            sg = oil_density/999.016
            api = ((141.5/sg)-131.5)
            sulfur = np.random.uniform(1.2, 2.8, n_samples)
            fzp = np.random.uniform(1.2, 1.6, n_samples)
            steam = np.random.uniform(8, 14, n_samples)
            reflux = np.random.uniform(1.5, 3.0, n_samples)
            top_t = np.random.uniform(115, 135, n_samples)
            lago_t = np.random.uniform(340, 365, n_samples)

            # Simulated physical cuts (with COT sensitivity)
            y_offgas = 0.02 + 0.0003 * (cot - 360) + np.random.normal(0, 0.002, n_samples)
            y_naphtha = 0.16 + 0.0005 * (cot - 360) + np.random.normal(0, 0.005, n_samples)
            y_kero = 0.12 + 0.0002 * (cot - 360) + np.random.normal(0, 0.004, n_samples)
            y_lago = 0.28 + 0.0012 * (cot - 360) + np.random.normal(0, 0.006, n_samples)
            y_residue = 1.0 - (y_offgas + y_naphtha + y_kero + y_lago)

            df = pd.DataFrame({
                'crude_flow': crude_flow, 'crude_api': api, 'sulfur_wt_pct': sulfur,
                'cot_degC': cot, 'flash_zone_p_kgcm2': fzp, 'stripping_steam_flow': steam,
                'reflux_ratio': reflux, 'top_temp_degC': top_t, 'lcgo_d86_95_degC': lago_t,
                'flow_offgas': y_offgas * crude_flow,
                'flow_naphtha': y_naphtha * crude_flow,
                'flow_kero': y_kero * crude_flow,
                'flow_lcgo': y_lago * crude_flow,
                'flow_residue': y_residue * crude_flow
            })

        st.subheader("Data Preview")
        st.dataframe(df.head(5))

        st.subheader("Column Mapping")
        c1, c2 = st.columns(2)
        feature_cols = c1.multiselect("Input Features (X)", options=list(df.columns), default=[c for c in DEFAULT_FEATURES if c in df.columns])
        target_cols = c2.multiselect("Product Outlet Flows (Y)", options=list(df.columns), default=[c for c in DEFAULT_TARGETS if c in df.columns])
        crude_flow_col = c1.selectbox("Crude Inlet Flow Column", options=list(df.columns), index=list(df.columns).index('crude_flow') if 'crude_flow' in df.columns else 0)

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Reconciling mass balances and training model..."):
                # 1. Mass Reconciliation Filter (<3% error)
                total_out = df[target_cols].sum(axis=1)
                imbalance = np.abs(total_out - df[crude_flow_col]) / df[crude_flow_col]
                valid_df = df[imbalance < 0.03].copy()

                # 2. Compute Yield Fractions (y_i = flow_i / crude_in)
                yield_targets = valid_df[target_cols].div(valid_df[crude_flow_col], axis=0)
                
                X = valid_df[feature_cols]
                y = yield_targets

                X_train, X_test, y_train, y_test, crude_train, crude_test = train_test_split(
                    X, y, valid_df[crude_flow_col], test_size=0.2, random_state=42
                )

                # 3. Fit Scaler & Regressor
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                base_reg = LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
                multi_model = MultiOutputRegressor(base_reg)
                multi_model.fit(X_train_scaled, y_train)

                # 4. Evaluate Test Metrics
                raw_yield_preds = multi_model.predict(X_test_scaled)
                # Normalize yields to sum to 1.0 (Mass Conservation)
                norm_yield_preds = raw_yield_preds / raw_yield_preds.sum(axis=1, keepdims=True)
                pred_flows = norm_yield_preds * crude_test.values.reshape(-1, 1)
                actual_flows = (y_test.values * crude_test.values.reshape(-1, 1))

                # Save Artifacts
                pipeline = {
                    "model": multi_model,
                    "scaler": scaler,
                    "features": feature_cols,
                    "targets": target_cols,
                    "crude_col": crude_flow_col
                }
                joblib.dump(pipeline, MODEL_FILE)
                st.success("Model trained and saved successfully!")

                # Display Metrics
                st.subheader("📊 Performance Metrics on Test Set")
                metrics = []
                for i, col in enumerate(target_cols):
                    r2 = r2_score(actual_flows[:, i], pred_flows[:, i])
                    mae = mean_absolute_error(actual_flows[:, i], pred_flows[:, i])
                    metrics.append({"Cut": col, "R² Score": f"{r2:.4f}", "MAE (t/h)": f"{mae:.2f}"})
                st.table(pd.DataFrame(metrics))

# ==============================================================================
# PAGE 2: PREDICTION INTERFACE
# ==============================================================================
elif page == "2. Real-Time Yield Prediction":
    st.header("🎯 CDU Cut Flow & Recovery Predictor")

    if not os.path.exists(MODEL_FILE):
        st.warning("⚠️ No trained model found. Please train the model on the '1. Model Training' page first.")
    else:
        pipeline = joblib.load(MODEL_FILE)
        features = pipeline["features"]
        targets = pipeline["targets"]

        st.subheader("Set Operating Parameters & Assay Properties")
        
        # Build interactive inputs dynamically
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            col = cols[i % 3]
            # Default values heuristic
            default_val = 360.0 if "cot" in feat.lower() else (400.0 if "flow" in feat.lower() else 30.0)
            input_data[feat] = col.number_input(f"{feat}", value=float(default_val), format="%.2f")

        if st.button("🔮 Predict Outlet Flows", type="primary"):
            input_df = pd.DataFrame([input_data])
            scaled_input = pipeline["scaler"].transform(input_df)

            # Predict yields and enforce mass conservation
            raw_yields = pipeline["model"].predict(scaled_input)[0]
            raw_yields = np.clip(raw_yields, 0, 1)
            norm_yields = raw_yields / np.sum(raw_yields)

            crude_in = input_data[pipeline["crude_col"]]
            pred_flows = norm_yields * crude_in

            # Results Display
            st.divider()
            st.subheader("📈 Predicted Outlet Yields & Rates")

            res_df = pd.DataFrame({
                "Product Stream": targets,
                "Yield (wt%)": [f"{y*100:.2f}%" for y in norm_yields],
                "Flow Rate (t/h)": [f"{f:.2f}" for f in pred_flows]
            })

            col_res, col_bal = st.columns([2, 1])
            col_res.table(res_df)

            with col_bal:
                st.metric("Total Crude In", f"{crude_in:.2f} t/h")
                st.metric("Sum of Product Flows", f"{np.sum(pred_flows):.2f} t/h")
                st.metric("Mass Closure Error", f"{abs(crude_in - np.sum(pred_flows)):.4f} t/h (0.00%)")
