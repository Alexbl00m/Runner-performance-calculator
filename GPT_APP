# Streamlit-runner-performance.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Runner Performance Calculator", layout="wide")

# LOGOTYPE
st.sidebar.image("Logotype_Light@2x.png", use_column_width=True)
st.sidebar.title("🏃‍♂️ Runner Performance")

# Helper: Time conversions
def parse_time(time_str):
    try:
        parts = [int(p) for p in time_str.strip().split(':')]
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
        else:
            return int(parts[0])
    except:
        st.error("Format as H:MM:SS or MM:SS")
        return 0

def seconds_to_str(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"

# Riegel prediction
def riegel(known_d, known_t, target_d, fatigue_factor=1.06):
    return known_t * (target_d / known_d) ** fatigue_factor

# Daniels VO2Max method
def daniels_vo2(time_sec, distance_km):
    v = distance_km / (time_sec / 60)  # km/min
    vo2 = -4.60 + 0.182258 * v * 60 + 0.000104 * (v * 60) ** 2
    return vo2

# Multi-point fatigue fitting (linear log-log)
def fit_fatigue(distances, times):
    log_d = np.log(distances)
    log_t = np.log(times)
    b, a = np.polyfit(log_d, log_t, 1)
    return np.exp(a), b

st.header("🏅 Race Time Predictor with Fatigue Curve")

st.info("Enter 2-3 race results. Predictions use Riegel's or curve-fit fatigue model.")

col1, col2, col3 = st.columns(3)

data = []
for col, label in zip([col1, col2, col3], ["First", "Second", "Third"]):
    with col:
        st.subheader(f"{label} Effort")
        dist = st.number_input(f"{label} Distance (km)", min_value=0.2, max_value=42.195, value=5.0)
        time = st.text_input(f"{label} Time (H:MM:SS or MM:SS)", value="25:00")
        sec = parse_time(time)
        if dist > 0 and sec > 0:
            data.append((dist, sec))

if len(data) < 2:
    st.warning("Input at least two valid efforts.")
    st.stop()

# Fit fatigue curve
D = np.array([d[0] for d in data])
T = np.array([d[1] for d in data])
A, fatigue_factor = fit_fatigue(D, T)

st.success(f"Estimated fatigue exponent b = {fatigue_factor:.3f} (Typical Riegel 1.06)")

# Prediction target
target_d = st.number_input("Target Distance (km)", min_value=0.4, max_value=42.195, value=21.0975)
pred_time = A * (target_d ** fatigue_factor)

st.markdown(f"**Predicted Time:** {seconds_to_str(pred_time)}")

# Optional Triathlon Brick Run Adjustment
if st.checkbox("Apply Triathlon Brick Run Adjustment (5% slower)"):
    tri_time = pred_time * 1.05
    st.markdown(f"**Brick Run Adjusted Time:** {seconds_to_str(tri_time)}")

# VO2Max estimation (Daniels)
vo2_estimates = [daniels_vo2(t, d) for d, t in zip(D, T)]
vo2_avg = np.mean(vo2_estimates)
st.info(f"Estimated VO₂max (Daniels method): {vo2_avg:.1f} ml/kg/min")

# Fatigue curve plot
st.subheader("Fatigue Curve")
curve_d = np.linspace(0.4, 42.195, 100)
curve_t = A * (curve_d ** fatigue_factor)

fig = go.Figure()
fig.add_trace(go.Scatter(x=curve_d, y=curve_t/60, mode='lines', name='Fatigue Curve'))
fig.add_trace(go.Scatter(x=D, y=T/60, mode='markers+text', text=[f"{d}km" for d in D],
                         marker=dict(size=12, color='red'), name='Your Efforts'))
fig.update_layout(xaxis_title='Distance (km)', yaxis_title='Time (min)', template='plotly_white')

st.plotly_chart(fig, use_container_width=True)

# Scientific explanation section
with st.expander("🔬 About the Model & Science"):
    st.markdown("""
    **Model Explanation:**
    - Predictions use a power-law fatigue model: `Time = A * Distance^b`
    - Default fatigue factor (b) ~1.06 per Riegel, adjusted here via your data

    **VO₂max Estimation:**
    - Based on Jack Daniels' Running Formula: `VO₂ = -4.6 + 0.182258 * v + 0.000104 * v²`

    **Fatigue Curve:**
    - Visualizes your sustainable pace decline with distance due to physiological fatigue

    **Triathlon Brick Run:**
    - Adds a 5% fatigue penalty reflecting the effect of cycling fatigue on running

    **Tip:**
    - Fatigue exponent >1.06 may indicate speed bias (sprinter) or undertraining endurance
    - <1.06 could indicate strong endurance adaptation
    """)

st.caption("Powered by Lindblom Coaching - Scientifically backed running analysis.")
