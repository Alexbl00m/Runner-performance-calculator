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

# Setup page configuration
st.set_page_config(
    page_title="Runner Performance Calculator",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (matching your brand)
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

# Function to load and display logo
def setup_logo():
    """Setup and display the logo"""
    try:
        if os.path.exists("logo.png"):
            logo = Image.open("logo.png")
            return logo
        else:
            # Create a basic placeholder logo
            img = Image.new('RGBA', (400, 200), color=(255, 255, 255, 0))
            return img
    except Exception as e:
        st.warning(f"Could not load or create logo: {e}")
        return None

# Utility functions for pace calculations
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
    """Main application function"""
    # Setup sidebar
    logo = setup_logo()
    if logo:
        st.sidebar.image(logo, width=200)
    
    st.sidebar.title("Runner Performance Calculator")
    st.sidebar.markdown("Predict race times and training zones based on your previous performances")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Race Predictor", "Training Zones", "About"])
    
    with tab1:
        st.header("Race Time Predictor")
        st.markdown("""
        <div class="result-box">
        Enter your best efforts for different distances to predict your times for common race distances and calculate your optimal training zones.
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for effort inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("First Effort")
            distance1_options = ["400m", "800m", "1000m", "1500m", "1 mile", "3000m", "5000m"]
            distance1_choice = st.selectbox("Distance", distance1_options, key="distance1")
            
            # Convert distance choice to kilometers
            distance1_km = 0.4  # Default 400m
            if distance1_choice == "800m":
                distance1_km = 0.8
            elif distance1_choice == "1000m":
                distance1_km = 1.0
            elif distance1_choice == "1500m":
                distance1_km = 1.5
            elif distance1_choice == "1 mile":
                distance1_km = 1.60934
            elif distance1_choice == "3000m":
                distance1_km = 3.0
            elif distance1_choice == "5000m":
                distance1_km = 5.0
            
            time1 = st.text_input("Time (MM:SS or H:MM:SS)", "3:30", key="time1")
            time1_seconds = parse_time_to_seconds(time1)
            
            # Display pace
            pace1_km, pace1_mile = calculate_pace(distance1_km, time1_seconds)
            st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            <span style="font-weight: bold;">Pace:</span> {pace1_km} min/km | {pace1_mile} min/mile
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Second Effort")
            distance2_options = ["400m", "800m", "1000m", "1500m", "1 mile", "3000m", "5000m", "10000m", "Half Marathon", "Marathon"]
            distance2_choice = st.selectbox("Distance", distance2_options, index=5, key="distance2")
            
            # Convert distance choice to kilometers
            distance2_km = 3.0  # Default 3000m
            if distance2_choice == "400m":
                distance2_km = 0.4
            elif distance2_choice == "800m":
                distance2_km = 0.8
            elif distance2_choice == "1000m":
                distance2_km = 1.0
            elif distance2_choice == "1500m":
                distance2_km = 1.5
            elif distance2_choice == "1 mile":
                distance2_km = 1.60934
            elif distance2_choice == "5000m":
                distance2_km = 5.0
            elif distance2_choice == "10000m":
                distance2_km = 10.0
            elif distance2_choice == "Half Marathon":
                distance2_km = 21.0975
            elif distance2_choice == "Marathon":
                distance2_km = 42.195
            
            time2 = st.text_input("Time (MM:SS or H:MM:SS)", "12:30", key="time2")
            time2_seconds = parse_time_to_seconds(time2)
            
            # Display pace
            pace2_km, pace2_mile = calculate_pace(distance2_km, time2_seconds)
            st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            <span style="font-weight: bold;">Pace:</span> {pace2_km} min/km | {pace2_mile} min/mile
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.subheader("Third Effort (Optional)")
            distance3_options = ["None", "400m", "800m", "1000m", "1500m", "1 mile", "3000m", "5000m", "10000m", "Half Marathon", "Marathon"]
            distance3_choice = st.selectbox("Distance", distance3_options, index=8, key="distance3")
            
            # Convert distance choice to kilometers
            distance3_km = 0  # Default None
            if distance3_choice == "400m":
                distance3_km = 0.4
            elif distance3_choice == "800m":
                distance3_km = 0.8
            elif distance3_choice == "1000m":
                distance3_km = 1.0
            elif distance3_choice == "1500m":
                distance3_km = 1.5
            elif distance3_choice == "1 mile":
                distance3_km = 1.60934
            elif distance3_choice == "3000m":
                distance3_km = 3.0
            elif distance3_choice == "5000m":
                distance3_km = 5.0
            elif distance3_choice == "10000m":
                distance3_km = 10.0
            elif distance3_choice == "Half Marathon":
                distance3_km = 21.0975
            elif distance3_choice == "Marathon":
                distance3_km = 42.195
            
            time3 = st.text_input("Time (MM:SS or H:MM:SS)", "45:00", key="time3")
            time3_seconds = parse_time_to_seconds(time3) if distance3_choice != "None" else 0
            
            # Display pace
            if distance3_choice != "None":
                pace3_km, pace3_mile = calculate_pace(distance3_km, time3_seconds)
                st.markdown(f"""
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <span style="font-weight: bold;">Pace:</span> {pace3_km} min/km | {pace3_mile} min/mile
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Target race selection
        st.subheader("Predict Time For")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            race_options = ["400m", "800m", "1000m", "1500m", "1 mile", "3000m", "5000m", "10000m", "Half Marathon", "Marathon"]
            target_race = st.selectbox("Race Distance", race_options, index=9)
            
            # Convert target race to kilometers
            target_km = 42.195  # Default Marathon
            if target_race == "400m":
                target_km = 0.4
            elif target_race == "800m":
                target_km = 0.8
            elif target_race == "1000m":
                target_km = 1.0
            elif target_race == "1500m":
                target_km = 1.5
            elif target_race == "1 mile":
                target_km = 1.60934
            elif target_race == "3000m":
                target_km = 3.0
            elif target_race == "5000m":
                target_km = 5.0
            elif target_race == "10000m":
                target_km = 10.0
            elif target_race == "Half Marathon":
                target_km = 21.0975
        
        # Calculate prediction
        distances = [distance1_km, distance2_km]
        times = [time1_seconds, time2_seconds]
        
        if distance3_km > 0 and time3_seconds > 0:
            distances.append(distance3_km)
            times.append(time3_seconds)
        
        # Calculate race predictions
        if all(d > 0 for d in distances) and all(t > 0 for t in times):
            # Use the Cameron method if there are at least two data points
            prediction_func, fatigue_factor = cameron_race_prediction(times, distances)
            predicted_time_seconds = prediction_func(target_km)
            
            # Calculate pace
            predicted_pace_km, predicted_pace_mile = calculate_pace(target_km, predicted_time_seconds)
            
            # Display prediction
            st.markdown("""
            <div class="result-box" style="margin-top: 30px; text-align: center;">
                <h2 style="color: #E6754E;">Predicted Race Time</h2>
                <div style="font-size: 28px; font-weight: bold; margin-bottom: 10px;">
                    {0}
                </div>
                <div style="font-size: 18px; margin-bottom: 20px;">
                    Pace: {1} min/km | {2} min/mile
                </div>
                <div style="font-size: 14px; color: #666;">
                    Calculated using fatigue factor: {3:.3f}
                </div>
            </div>
            """.format(
                seconds_to_time_str(predicted_time_seconds),
                predicted_pace_km,
                predicted_pace_mile,
                fatigue_factor
            ), unsafe_allow_html=True)
            
            # Calculate triathlon run adjustment (5% slower for brick runs)
            if target_race in ["5000m", "10000m", "Half Marathon", "Marathon"]:
                tri_adjustment = predicted_time_seconds * 0.05
                tri_adjusted_time = predicted_time_seconds + tri_adjustment
                tri_pace_km, tri_pace_mile = calculate_pace(target_km, tri_adjusted_time)
                
                st.markdown("""
                <div class="result-box" style="text-align: center;">
                    <h3 style="color: #E6754E;">Triathlon Brick Run Adjustment (5% slower)</h3>
                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">
                        {0}
                    </div>
                    <div style="font-size: 16px;">
                        Pace: {1} min/km | {2} min/mile
                    </div>
                </div>
                """.format(
                    seconds_to_time_str(tri_adjusted_time),
                    tri_pace_km,
                    tri_pace_mile
                ), unsafe_allow_html=True)
            
            # Predict other common race distances and create comparison table
            race_distances = {
                "400m": 0.4,
                "800m": 0.8,
                "1500m": 1.5,
                "1 mile": 1.60934,
                "5K": 5.0,
                "10K": 10.0,
                "Half Marathon": 21.0975,
                "Marathon": 42.195
            }
            
            # Only include races we don't already have actuals for
            existing_distances = set(distances)
            comparison_data = []
            
            for race_name, race_dist in race_distances.items():
                # Skip if we already have an actual time for this distance
                if race_dist in existing_distances:
                    actual_idx = distances.index(race_dist)
                    actual_time = seconds_to_time_str(times[actual_idx])
                    actual_pace_km, actual_pace_mile = calculate_pace(race_dist, times[actual_idx])
                    
                    comparison_data.append({
                        "Race": race_name,
                        "Distance (km)": f"{race_dist:.2f}",
                        "Predicted": seconds_to_time_str(prediction_func(race_dist)),
                        "Actual": actual_time,
                        "Predicted Pace": f"{calculate_pace(race_dist, prediction_func(race_dist))[0]} min/km",
                        "Actual Pace": f"{actual_pace_km} min/km",
                        "Difference": "0:00"
                    })
                else:
                    comparison_data.append({
                        "Race": race_name,
                        "Distance (km)": f"{race_dist:.2f}",
                        "Predicted": seconds_to_time_str(prediction_func(race_dist)),
                        "Actual": "—",
                        "Predicted Pace": f"{calculate_pace(race_dist, prediction_func(race_dist))[0]} min/km",
                        "Actual Pace": "—",
                        "Difference": "—"
                    })
            
            # Show comparative predictions
            st.markdown("### Race Time Comparisons")
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            # Plot prediction curve
            st.markdown("### Race Prediction Curve")
            
            # Create log-spaced distances from 400m to Marathon
            distances_to_plot = np.logspace(np.log10(0.4), np.log10(42.195), 100)
            predicted_times = [prediction_func(d) / 60 for d in distances_to_plot]  # Convert to minutes
            
            # Mark actual efforts on the plot
            actual_distances = distances
            actual_times = [t / 60 for t in times]  # Convert to minutes
            
            # Create plot
            fig = px.line(
                x=distances_to_plot, 
                y=predicted_times,
                labels={"x": "Distance (km)", "y": "Time (minutes)"},
                log_x=True,
                title="Race Prediction Power Curve"
            )
            
            # Add markers for actual performances
            fig.add_scatter(
                x=actual_distances, 
                y=actual_times, 
                mode='markers+text',
                marker=dict(color='red', size=12),
                text=[f"{d:.1f}km" for d in actual_distances],
                textposition="top center",
                name="Your Performances"
            )
            
            # Mark the target race
            target_time = prediction_func(target_km) / 60
            fig.add_scatter(
                x=[target_km], 
                y=[target_time], 
                mode='markers+text',
                marker=dict(color='green', size=12, symbol='star'),
                text=[f"{target_race}"],
                textposition="top center",
                name="Target Race"
            )
            
            # Customize layout
            fig.update_layout(
                template="plotly_white",
                title_font_size=20,
                title_font_family="Montserrat",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.header("Training Zones Calculator")
        st.markdown("""
        <div class="result-box">
        View your optimal training zones based on your running performances. These zones are calculated using the Yousli method, which uses your threshold pace as a reference.
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate threshold pace from the most relevant distance
        # Ideally we use the 5K-10K range as a good indicator of threshold pace
        threshold_pace_seconds = None
        threshold_distance = None
        
        if len(distances) > 0 and len(times) > 0:
            # Find the closest race to 5K-10K range as threshold indicator
            distances_array = np.array(distances)
            threshold_idx = np.argmin(np.abs(distances_array - 5.0))
            threshold_distance = distances[threshold_idx]
            threshold_time = times[threshold_idx]
            
            # Calculate threshold pace in seconds per km
            threshold_pace_seconds = threshold_time / threshold_distance
            
            # Determine the reference race used for zones
            if threshold_distance < 3.0:
                reference_race = f"{int(threshold_distance * 1000)}m"
            elif threshold_distance == 3.0:
                reference_race = "3000m"
            elif threshold_distance == 5.0:
                reference_race = "5000m"
            elif threshold_distance == 10.0:
                reference_race = "10000m"
            elif threshold_distance == 21.0975:
                reference_race = "Half Marathon"
            elif threshold_distance == 42.195:
                reference_race = "Marathon"
            else:
                reference_race = f"{threshold_distance:.1f}km"
            
            # Display threshold pace
            threshold_pace_km, threshold_pace_mile = calculate_pace(1.0, threshold_pace_seconds)
            
            st.markdown(f"""
            <div class="result-box" style="text-align: center;">
                <h3>Reference Threshold Pace</h3>
                <p>Based on your {reference_race} performance</p>
                <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                    {threshold_pace_km} min/km | {threshold_pace_mile} min/mile
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate Yousli running zones
            zones = calculate_running_zones(threshold_pace_seconds)
            
            st.markdown("""
            <h3 style="margin-top: 30px;">Yousli Running Zones</h3>
            <p>These zones are based on your threshold pace and can be used to guide your training.</p>
            """, unsafe_allow_html=True)
            
            # Display the zones in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                for zone_name in ["Easy", "Zone 2", "Endurance", "Threshold"]:
                    zone = zones[zone_name]
                    min_pace_km = format_pace(zone["min"])
                    max_pace_km = "slower" if zone["max"] == float('inf') else format_pace(zone["max"])
                    
                    min_pace_sec_mile = zone["min"] * 1.60934
                    max_pace_sec_mile = float('inf') if zone["max"] == float('inf') else zone["max"] * 1.60934
                    min_pace_mile = format_pace(min_pace_sec_mile)
                    max_pace_mile = "slower" if max_pace_sec_mile == float('inf') else format_pace(max_pace_sec_mile)
                    
                    st.markdown(f"""
                    <div class="result-box" style="margin-bottom: 15px;">
                        <div class="zone-header">{zone_name}:</div>
                        <div class="zone-pace">
                            {min_pace_km} - {max_pace_km if max_pace_km != "slower" else "slower"} min/km
                        </div>
                        <div class="zone-pace" style="font-size: 16px; color: #666;">
                            {min_pace_mile} - {max_pace_mile if max_pace_mile != "slower" else "slower"} min/mile
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                for zone_name in ["Suprathreshold", "VO2max", "Power"]:
                    zone = zones[zone_name]
                    min_pace_km = "faster" if zone["min"] == 0 else format_pace(zone["min"])
                    max_pace_km = format_pace(zone["max"])
                    
                    min_pace_sec_mile = 0 if zone["min"] == 0 else zone["min"] * 1.60934
                    max_pace_sec_mile = zone["max"] * 1.60934
                    min_pace_mile = "faster" if min_pace_sec_mile == 0 else format_pace(min_pace_sec_mile)
                    max_pace_mile = format_pace(max_pace_sec_mile)
                    
                    st.markdown(f"""
                    <div class="result-box" style="margin-bottom: 15px;">
                        <div class="zone-header">{zone_name}:</div>
                        <div class="zone-pace">
                            {min_pace_km if min_pace_km != "faster" else "faster"} - {max_pace_km} min/km
                        </div>
                        <div class="zone-pace" style="font-size: 16px; color: #
