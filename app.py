import streamlit as st
import serial
import pandas as pd
import numpy as np
import joblib
import time

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")

PORT="/dev/cu.usbserial-0001"
BAUD=115200

st.title("🏗️ SENTINEL-AI Structural Monitor (GNN-Inspired)")

mode = st.radio("Mode",["Live ESP32","Simulation"])

chart_placeholder = st.empty()
status_box = st.empty()
risk_box = st.empty()
conn_box = st.empty()

live=False

# ===== SERIAL CONNECTION =====
if mode=="Live ESP32":
    try:
        ser = serial.Serial(PORT,BAUD,timeout=1)
        time.sleep(2)
        live=True
        conn_box.success("✅ ESP32 Connected")
    except:
        conn_box.error("❌ ESP32 not found")

# ===== SENSOR READ =====
def read_vibration():
    if live:
        try:
            line=ser.readline().decode().strip()
            return float(line) if line else 0.0
        except:
            return 0.0
    else:
        return np.random.choice([
            np.random.uniform(0,0.1),
            np.random.uniform(0.2,1.5),
            np.random.uniform(2,8)
        ])

# ===== DATAFRAME =====
data=pd.DataFrame(columns=["Vibration"])

buffer=[]
history=[]
WINDOW=3

# ===== LOOP =====
for _ in range(10000):

    vib = read_vibration()

    # noise filter
    if vib < 0.02:
        vib = 0

    buffer.append(vib)
    if len(buffer)>WINDOW:
        buffer.pop(0)

    # =========================
    # GNN-STYLE NODE FUSION
    # =========================

    # Node 1 — vibration
    vibration_node = np.mean(buffer)

    # Node 2 — load proxy
    load_node = vibration_node * 0.8

    # Node 3 — history node
    history.append(vibration_node)
    if len(history)>5:
        history.pop(0)

    history_node = np.mean(history)

    # Graph aggregation (message passing style)
    feature = (
        vibration_node*0.5 +
        load_node*0.3 +
        history_node*0.2
    )

    # ===== ML PREDICTION =====
    pred=model.predict([[feature]])[0]

    # ===== CHART =====
    data.loc[len(data)] = [feature]
    chart_placeholder.line_chart(data)

    # ===== RISK =====
    risk=min(feature*15,100)
    risk_box.metric("Risk Level %",f"{risk:.1f}")

    # ===== STATUS =====
    if pred=="damage":
        status_box.error("🔴 DAMAGE")
    elif pred=="load":
        status_box.warning("🟡 LOAD")
    else:
        status_box.success("🟢 NORMAL")

    time.sleep(0.07)