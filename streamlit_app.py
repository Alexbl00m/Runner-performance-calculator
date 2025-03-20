import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import math
import io
from PIL import Image
import os

# Import the logo setup function
from logo_setup import create_logo

# Constants and utility functions
def format_time_input(time_str):
    """Format time string for input validation"""
    if ":" not in time_str:
        # Assume seconds only
        try:
            seconds = float(time_str)
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}:{seconds:02d}"
        except:
            return time_str
    return time_str

def parse_time_to_seconds(time_str):
    """Convert time string (H:MM:SS or MM:SS) to seconds"""
    if not time_str:
        return 0
    
    parts = time_str.strip().split(':')
    if len(parts) == 3:  # H:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    else:
        try:
            # Try to convert directly to seconds
            return int(float(time_str))
        except:
            st.error(f"Invalid time format: {time_str}. Please use H:MM:SS or MM:SS.")
            return 0

def seconds_to_time_str(seconds):
    """Convert seconds to time string (H:MM:SS or MM:SS)"""
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

def calculate_pace(distance_km, time_seconds):
    """Calculate pace in min/km and min/mile"""
    if distance_km <= 0 or time_seconds <= 0:
        return "0:00", "0:00"
    
    # Min per km
    pace_sec_per_km = time_seconds / distance_km
    pace_min_per_km = pace_sec_per_km / 60
    pace_min = int(pace_min_per_km)
    pace_sec = int((pace_min_per_km - pace_min) * 60)
    pace_km = f"{pace_min}:{pace_sec:02d}"
    
    # Min per mile (1 mile = 1.60934 km)
    pace_sec_per_mile = time_seconds / (distance_km / 1.60934)
    pace_min_per_mile = pace_sec_per_mile / 60
    pace_min = int(pace_min_per_mile)
    pace_sec = int((pace_min_per_mile - pace_min) * 60)
    pace_mile = f"{pace_min}:{pace_sec:02d}"
    
    return pace_km, pace_mile

def format_pace(seconds_per_km):
    """Format pace from seconds per km to MM:SS string"""
    minutes = int(seconds_per_km / 60)
    seconds = int(seconds_per_km % 60)
    return f"{minutes}:{seconds:02d}"

def riegel_race_prediction(known_distance, known_time, target_distance, fatigue_factor=1.06):
    """
    Predict race time using Riegel's formula
    T2 = T1 × (D2/D1)^fatigue_factor
    """
    if known_distance <= 0 or known_time <= 0:
        return 0
    
    # Calculate predicted time in seconds
    predicted_time = known_time * ((target_distance / known_distance) ** fatigue_factor)
    return predicted_time

def cameron_race_prediction(times, distances):
    """
    Use Cameron's formula with multiple data points to predict race times
    Based on the curve fitting approach for a power-law relationship
    """
    if len(times) < 2 or len(distances) < 2:
        return lambda d: riegel_race_prediction(distances[0], times[0], d)
    
    # Take log of distances and times
    log_distances = np.log(distances)
    log_times = np.log(times)
    
    # Find best fit for T = a * D^b
    coeffs = np.polyfit(log_distances, log_times, 1)
    b = coeffs[0]  # Fatigue factor
    a = np.exp(coeffs[1])  # Scaling factor
    
    # Return a function that predicts time for any distance
    def predict(distance):
        return a * (distance ** b)
    
    return predict, b

def calculate_running_zones(threshold_pace_sec):
    """Calculate Yousli running zones based on threshold pace in seconds per km"""
    zones = {
        "Easy": {"min": threshold_pace_sec * 1.5, "max": float('inf')},
        "Zone 2": {"min": threshold_pace_sec * 1.2, "max": threshold_pace_sec * 1.5},
        "Endurance": {"min": threshold_pace_sec * 1.1, "max": threshold_pace_sec * 1.2},
        "Threshold": {"min": threshold_pace_sec * 0.98, "max": threshold_pace_sec * 1.1},
        "Suprathreshold": {"min": threshold_pace_sec * 0.95, "max": threshold_pace_sec * 0.98},
        "VO2max": {"min": threshold_pace_sec * 0.85, "max": threshold_pace_sec * 0.95},
        "Power": {"min": 0, "max": threshold_pace_sec * 0.85}
    }
    return zones

def main():
    # Setup page configuration
    st.set_page_config(
        page_title="Runner Performance Calculator",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    .main {
        background-color: #FFFFFF;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        color: #E6754E;
    }

    .stButton>button {
        background-color: #E6754E;
        color: white;
        font-family: 'Montserrat', sans-serif;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }

    .stButton>button:hover {
        background-color: #c45d3a;
    }

    .highlight {
        color: #E6754E;
        font-weight: 600;
    }

    .result-box {
        background-color: #f8f8f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E6754E;
        margin-bottom: 20px;
    }

    .zone-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #333;
    }

    .zone-pace {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #E6754E;
    }

    .zone-description {
        font-size: 14px;
        color: #666;
        margin-bottom: 0;
    }

    footer {
        font-family: 'Montserrat', sans-serif;
        font-size: 12px;
        color: #888888;
        text-align: center;
        margin-top: 50px;
    }

    /* Custom divider */
    .custom-divider {
        height: 3px;
        background-color: #E6754E;
        margin: 20px 0;
        border-radius: 2px;
    }

    /* Streamlit component overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f5f5f5;
        border-radius: 5px 5px 0 0;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #E6754E !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create and display logo
    logo_path = create_logo()
    
    # Setup sidebar
    if logo_path:
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=200)
    
    st.sidebar.title("Runner Performance Calculator")
    st.sidebar.markdown("Predict race times and training zones based on your previous performances")
    
    # Import the runner calculator code
    from runner_calculator import main as run_calculator
    
    # Run the calculator
    run_calculator()

if __name__ == "__main__":
    main()
