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

# ==============================================================================
# DATABASE SETUP & ACCESS CONTROL LAYER
# ==============================================================================
DB_PATH = "audit_telemetry.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            role TEXT
        )
    ''')
    
    # 2. Access & Geolocation Logs
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            login_time TIMESTAMP,
            ip_address TEXT,
            city TEXT,
            region TEXT,
            country TEXT,
            duration_minutes REAL
        )
    ''')

    # 3. Model Registry per User
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            model_tag TEXT,
            model_path TEXT,
            training_rows INTEGER,
            created_at TIMESTAMP
        )
    ''')

    # 4. Input & Prediction Telemetry
    c.execute('''
        CREATE TABLE IF NOT EXISTS simulation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TIMESTAMP,
            inputs_json TEXT,
            predictions_json TEXT
        )
    ''')
    
    # Create default Admin and Demo User if table is empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        admin_pw = hashlib.sha256("Admin@123".encode()).hexdigest()
        user_pw = hashlib.sha256("User@123".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("admin", admin_pw, "admin"))
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("engineer1", user_pw, "user"))
    
    conn.commit()
    conn.close()

init_db()

# --- Utility Functions: Auth & Geolocation ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", (username, hash_password(password)))
    row = c.fetchone()
    conn.close()
    return row["role"] if row else None

def get_visitor_geo():
    """Captures public IP address and geolocation via ipapi.co."""
    try:
        res = requests.get("https://ipapi.co/json/", timeout=3.0).json()
        return {
            "ip": res.get("ip", "Local/Unknown"),
            "city": res.get("city", "Unknown"),
            "region": res.get("region", "Unknown"),
            "country": res.get("country_name", "Unknown")
        }
    except Exception:
        return {"ip": "127.0.0.1", "city": "Local", "region": "Local", "country": "Local"}

def log_session_start(username):
    geo = get_visitor_geo()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO access_logs (username, login_time, ip_address, city, region, country, duration_minutes)
        VALUES (?, ?, ?, ?, ?, ?, 0.0)
    ''', (username, datetime.now(), geo["ip"], geo["city"], geo["region"], geo["country"]))
    conn.commit()
    log_id = c.lastrowid
    conn.close()
    return log_id

def log_simulation(username, inputs, outputs):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO simulation_logs (username, timestamp, inputs_json, predictions_json)
        VALUES (?, ?, ?, ?)
    ''', (username, datetime.now(), json.dumps(inputs), json.dumps(outputs)))
    conn.commit()
    conn.close()

# ==============================================================================
# UI AUTHENTICATION INTERFACE
# ==============================================================================
st.set_page_config(page_title="CDU Twin Enterprise", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None
    st.session_state["login_time"] = None
    st.session_state["session_log_id"] = None

if not st.session_state["authenticated"]:
    st.title("🛢️ Refinery CDU Digital Twin - Enterprise Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
        
        if submitted:
            role = verify_login(u, p)
            if role:
                st.session_state["authenticated"] = True
                st.session_state["username"] = u
                st.session_state["role"] = role
                st.session_state["login_time"] = datetime.now()
                st.session_state["session_log_id"] = log_session_start(u)
                st.success(f"Welcome, {u} ({role.upper()})")
                st.rerun()
            else:
                st.error("Invalid Username or Password")
    st.stop()

# --- Sidebar Management & Dynamic Role Views ---
st.sidebar.markdown(f"**Logged in as:** `{st.session_state['username']}` | **Role:** `{st.session_state['role'].upper()}`")
session_mins = round((datetime.now() - st.session_state["login_time"]).total_seconds() / 60.0, 1)
st.sidebar.caption(f"Session Active: {session_mins} mins")

if st.sidebar.button("Logout"):
    # Update final session duration
    if st.session_state["session_log_id"]:
        conn = get_db_connection()
        conn.execute("UPDATE access_logs SET duration_minutes = ? WHERE id = ?", (session_mins, st.session_state["session_log_id"]))
        conn.commit()
        conn.close()
    st.session_state.clear()
    st.rerun()

# Build menu according to role
menu_options = ["1. Model Training & DCS Upload", "2. Real-Time Yield Prediction", "3. My Models"]
if st.session_state["role"] == "admin":
    menu_options.append("🛡️ Admin Audit & Global Telemetry")

page = st.sidebar.radio("Navigation", menu_options)

# ==============================================================================
# PAGE: REAL-TIME YIELD PREDICTION (WITH INPUT TELEMETRY)
# ==============================================================================
if page == "2. Real-Time Yield Prediction":
    st.header("🎯 Autonomous CDU Prediction")
    st.markdown("Enter boundary operating conditions. Every simulation run is logged for compliance.")

    col1, col2, col3 = st.columns(3)
    cot = col1.number_input("Furnace COT (°C)", value=358.5)
    fzp = col2.number_input("Flash Zone Pressure (kg/cm²)", value=1.45)
    feed = col3.number_input("Crude Flow (t/h)", value=410.0)

    if st.button("Run Simulation & Predict", type="primary"):
        # Prediction calculation placeholder
        inputs_payload = {"cot_degC": cot, "flash_zone_p_kgcm2": fzp, "crude_flow": feed}
        outputs_payload = {
            "flow_offgas": round(feed * 0.021, 2),
            "flow_naphtha": round(feed * 0.165, 2),
            "flow_kero": round(feed * 0.123, 2),
            "flow_lago": round(feed * 0.282, 2),
            "flow_residue": round(feed * 0.409, 2)
        }
        
        # Save exact user input and calculated yields to database
        log_simulation(st.session_state["username"], inputs_payload, outputs_payload)
        
        st.success("Prediction complete and execution telemetry archived.")
        st.write("### Predicted Product Mass Rates (t/h):")
        st.json(outputs_payload)

# ==============================================================================
# PAGE: MY MODELS (ROLE-FILTERED VIEW)
# ==============================================================================
elif page == "3. My Models":
    st.header(f"📦 Models Created by `{st.session_state['username']}`")
    conn = get_db_connection()
    
    # Regular users only see their own models
    my_models = pd.read_sql_query(
        "SELECT model_tag, training_rows, created_at FROM user_models WHERE username = ?",
        conn, params=(st.session_state['username'],)
    )
    conn.close()

    if my_models.empty:
        st.info("No custom models saved under your user account yet.")
    else:
        st.dataframe(my_models, use_container_width=True)

# ==============================================================================
# PAGE: ADMIN AUDIT & GLOBAL TELEMETRY (ADMIN EXCLUSIVE)
# ==============================================================================
elif page == "🛡️ Admin Audit & Global Telemetry":
    st.header("🛡️ Global Enterprise Telemetry & User Audit")
    conn = get_db_connection()

    tab_access, tab_sims, tab_all_models = st.tabs([
        "📍 Access, Location & Durations", 
        "📝 All Inputted Data Logs", 
        "🗂️ Global Models (All Users)"
    ])

    with tab_access:
        st.subheader("User Login Geolocation & Session Lengths")
        access_df = pd.read_sql_query("SELECT username, login_time, ip_address, city, region, country, duration_minutes FROM access_logs ORDER BY login_time DESC", conn)
        st.dataframe(access_df, use_container_width=True)

    with tab_sims:
        st.subheader("Historical Inputs & Predictions (Audit Trail)")
        sims_df = pd.read_sql_query("SELECT id, username, timestamp, inputs_json, predictions_json FROM simulation_logs ORDER BY timestamp DESC", conn)
        st.dataframe(sims_df, use_container_width=True)

    with tab_all_models:
        st.subheader("Master List of All Uploaded Models")
        all_models_df = pd.read_sql_query("SELECT * FROM user_models ORDER BY created_at DESC", conn)
        st.dataframe(all_models_df, use_container_width=True)

    conn.close()
