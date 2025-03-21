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
        return lambda d: riegel_race_prediction(distances[0], times[0], d), 1.06
    
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
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Race Predictor", "Training Zones", "Pace Calculator", "About"])
    
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
    
    with tab4:
        st.header("About the Runner Performance Calculator")
        
        st.markdown("""
        <div class="result-box">
        <h3>How This Calculator Works</h3>
        <p>The Runner Performance Calculator uses advanced mathematical models to predict your race times across different distances based on your previous performances. It employs a modified version of Peter Riegel's formula with an adaptive fatigue factor calculated from your input data.</p>
        
        <p>The basic equation used is:</p>
        <p style="font-style: italic; font-weight: bold; text-align: center; margin: 20px 0;">
            T2 = T1 × (D2/D1)<sup>f</sup>
        </p>
        
        <p>Where:</p>
        <ul>
            <li>T2 is the predicted time for distance D2</li>
            <li>T1 is your known time for distance D1</li>
            <li>f is the fatigue factor (typically between 1.05 and 1.15)</li>
        </ul>
        
        <p>When multiple performances are provided, the calculator employs curve fitting to determine the optimal fatigue factor specifically for you.</p>
        </div>
        
        <div class="result-box">
        <h3>Training Zones Methodology</h3>
        <p>The training zones are calculated using the Lindblom Coaching 7-zone system, which is based on your threshold pace. The threshold is determined by analyzing your performances with special emphasis on races in the 5K-10K range, which typically represent a good approximation of lactate threshold for most runners.</p>
        
        <p>These zones provide structured intensity guidelines for different types of workouts, helping you train more efficiently and reduce the risk of injury by ensuring you're running at the appropriate effort level for each training session.</p>
        </div>
        
        <div class="result-box">
        <h3>Tips for Accurate Predictions</h3>
        <ul>
            <li>Use recent race results or time trials for the most accurate predictions</li>
            <li>Include performances from different distances to improve the accuracy of the fatigue factor</li>
            <li>For best results, at least one of your performances should be from a distance similar to your target race</li>
            <li>Remember that predictions assume similar conditions (weather, terrain, elevation) and optimal race preparation</li>
        </ul>
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

        # Initialize variables with default values
        pace_seconds_per_km = 0
        race_distance_km = 0
        total_seconds = 0
        
        # If we have valid pace, calculate and display splits
        if pace_seconds_per_km > 0 and race_distance_km > 0:
            st.markdown("### Race Splits")
            
            # Determine if kilometer or mile splits are more appropriate
            if race_distance_km < 3:
                # For shorter races, use 400m splits
                split_distance = 0.4  # km
                split_label = "400m"
                num_splits = int(race_distance_km / split_distance)
                partial_split = race_distance_km % split_distance
            elif race_distance_km < 10:
                # For medium races, use kilometer splits
                split_distance = 1.0  # km
                split_label = "1 km"
                num_splits = int(race_distance_km)
                partial_split = race_distance_km - num_splits
            else:
                # For longer races, use mile splits
                split_distance = 1.60934  # km (1 mile)
                split_label = "1 mile"
                num_splits = int(race_distance_km / split_distance)
                partial_split = race_distance_km % split_distance
            
            # Calculate split times based on strategy
            splits_data = []
            halfway_point = num_splits / 2
            cumulative_time = 0
            
            for i in range(1, num_splits + 1):
                # For negative/positive splits, adjust based on position in race
                if pace_strategy == "Even Pace":
                    adjusted_pace = pace_seconds_per_km
                else:
                    if i <= halfway_point:
                        adjusted_pace = pace_seconds_per_km * (1 + split_factor)
                    else:
                        adjusted_pace = pace_seconds_per_km * (1 - split_factor)
                
                split_time = adjusted_pace * split_distance
                cumulative_time += split_time
                
                splits_data.append({
                    "Split": f"{i} ({split_label})",
                    "Distance": f"{i * split_distance:.2f} km",
                    "Split Time": seconds_to_time_str(split_time),
                    "Pace": format_pace(adjusted_pace),
                    "Cumulative Time": seconds_to_time_str(cumulative_time)
                })
            
            # Add partial split if needed
            if partial_split > 0.01:  # Only if it's significant
                # For the partial split, use the final pace adjustment
                if pace_strategy == "Even Pace":
                    adjusted_pace = pace_seconds_per_km
                else:
                    adjusted_pace = pace_seconds_per_km * (1 - split_factor)
                
                split_time = adjusted_pace * partial_split
                cumulative_time += split_time
                
                splits_data.append({
                    "Split": f"Final ({partial_split*1000:.0f}m)",
                    "Distance": f"{race_distance_km:.2f} km",
                    "Split Time": seconds_to_time_str(split_time),
                    "Pace": format_pace(adjusted_pace),
                    "Cumulative Time": seconds_to_time_str(cumulative_time)
                })
            
            # Display splits in a table
            splits_df = pd.DataFrame(splits_data)
            st.dataframe(splits_df, hide_index=True, use_container_width=True)
            
            # Create a visualization of split paces
            st.markdown("### Split Visualization")
            
            fig = px.bar(
                splits_data, 
                x="Split", 
                y=[format_pace(pace_seconds_per_km)]*len(splits_data), 
                title="Race Splits vs. Target Pace",
                labels={"y": "Pace (min/km)", "x": "Split"}
            )
            
            # Add a line for target pace
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(splits_data)-0.5,
                y0=format_pace(pace_seconds_per_km),
                y1=format_pace(pace_seconds_per_km),
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation for target pace
            fig.add_annotation(
                x=len(splits_data)-0.5,
                y=format_pace(pace_seconds_per_km),
                text=f"Target Pace: {format_pace(pace_seconds_per_km)} min/km",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
            
            # Customize layout
            fig.update_layout(
                template="plotly_white",
                height=400,
                xaxis_title="Race Segments",
                yaxis_title="Pace (min/km)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add tips for race execution
            st.markdown("""
            <div class="result-box">
                <h3>Race Execution Tips</h3>
                <ul>
                    <li><strong>Start conservatively</strong> - Many runners go out too fast and pay for it later</li>
                    <li><strong>Focus on effort</strong> - Pay attention to your perceived exertion rather than just the watch</li>
                    <li><strong>Adjust for conditions</strong> - Heat, wind, and hills will affect your pace; be flexible</li>
                    <li><strong>Fuel properly</strong> - For races over 75 minutes, ensure adequate carbohydrate intake</li>
                    <li><strong>Mental check-ins</strong> - Break the race into segments and focus on executing each segment well</li>
                </ul>
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
        
        if 'distances' in locals() and 'times' in locals() and len(distances) > 0 and len(times) > 0:
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
                        <div class="zone-pace" style="font-size: 16px; color: #666;">
                            {min_pace_mile if min_pace_mile != "faster" else "faster"} - {max_pace_mile} min/mile
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display zone descriptions
            st.markdown("""
            <h3 style="margin-top: 30px;">Training Zone Descriptions</h3>
            """, unsafe_allow_html=True)
            
            zone_descriptions = [
                {
                    "Zone": "Easy",
                    "Description": "Very easy, recovery runs. Builds base aerobic fitness and allows for recovery between harder sessions.",
                    "Effort": "Very light effort, could hold a full conversation easily",
                    "Use For": "Recovery days, warm-ups, cool-downs"
                },
                {
                    "Zone": "Zone 2",
                    "Description": "Light aerobic work that develops fat metabolism and cardiovascular efficiency.",
                    "Effort": "Comfortable pace, can hold a conversation",
                    "Use For": "Long runs, recovery runs, base building"
                },
                {
                    "Zone": "Endurance",
                    "Description": "Moderate effort that builds aerobic capacity. The pace you could maintain for a long time.",
                    "Effort": "Moderately challenging but sustainable, conversation becomes more difficult",
                    "Use For": "Longer tempo runs, marathon pace training"
                },
                {
                    "Zone": "Threshold",
                    "Description": "This is your lactate threshold pace, approximately your 10K-15K race pace.",
                    "Effort": "Comfortably hard, limited talking possible",
                    "Use For": "Tempo runs, threshold intervals (3-5 minute repeats)"
                },
                {
                    "Zone": "Suprathreshold",
                    "Description": "Just above threshold, roughly 5K race pace for trained runners.",
                    "Effort": "Hard but sustainable for 5K, breathing becomes labored",
                    "Use For": "5K-specific workouts, shorter intervals (2-3 minutes)"
                },
                {
                    "Zone": "VO2max",
                    "Description": "High-intensity effort that develops maximum aerobic capacity, similar to 1500m-3000m race pace.",
                    "Effort": "Very hard, breathing heavily, only a few words possible at a time",
                    "Use For": "VO2max intervals (2-5 minutes with equal recovery)"
                },
                {
                    "Zone": "Power",
                    "Description": "Maximum intensity, develops neuromuscular power and anaerobic capacity.",
                    "Effort": "All-out effort, not sustainable for more than 1-2 minutes",
                    "Use For": "Short intervals, sprints, strides (15s-90s with full recovery)"
                }
            ]
            
            # Display zone descriptions as a table
            zone_df = pd.DataFrame(zone_descriptions)
            st.dataframe(zone_df, hide_index=True, use_container_width=True)
        
        else:
            st.warning("Please enter at least one valid effort in the Race Predictor tab to calculate training zones.")
    
    with tab3:
        st.header("Race Pace Calculator")
        st.markdown("""
        <div class="result-box">
        Plan your race strategy by calculating split times and finish times. Enter your target pace or target time to get detailed lap breakdowns.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Race Information")
            
            # Race distance selection
            race_options = {
                "400m": 0.4,
                "800m": 0.8,
                "1000m": 1.0,
                "1500m": 1.5,
                "1 mile": 1.60934,
                "3000m": 3.0,
                "5K": 5.0,
                "10K": 10.0,
                "15K": 15.0,
                "Half Marathon": 21.0975,
                "Marathon": 42.195,
                "Custom Distance": 0
            }
            
            race_distance_choice = st.selectbox(
                "Race Distance", 
                list(race_options.keys()),
                index=6,  # Default to 5K
                key="pace_race_distance"
            )
            
            # If custom distance is selected, allow user input
            if race_distance_choice == "Custom Distance":
                custom_distance = st.number_input("Enter distance in kilometers:", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
                race_distance_km = custom_distance
            else:
                race_distance_km = race_options[race_distance_choice]
            
            # Calculate lap and mile count
            lap_count = race_distance_km / 0.4  # Standard track lap is 400m
            mile_count = race_distance_km / 1.60934
            
            # Display track laps and miles
            st.markdown(f"""
            <div style="background-color: #f8f8f8; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <span style="font-weight: bold;">Distance:</span> {race_distance_km} km
                <br><span style="font-weight: bold;">Track laps (400m):</span> {lap_count:.2f}
                <br><span style="font-weight: bold;">Miles:</span> {mile_count:.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Allow user to select pace strategy
            pace_strategy = st.radio(
                "Pacing Strategy",
                ["Even Pace", "Negative Split", "Positive Split"],
                horizontal=True
            )
            
            # Split settings based on strategy
            if pace_strategy == "Even Pace":
                st.info("Even pace: maintain the same pace throughout the entire race.")
                split_factor = 0.0
            elif pace_strategy == "Negative Split":
                split_factor = st.slider("First Half Adjustment:", min_value=0.01, max_value=0.10, value=0.05, step=0.01, format="+%g")
                st.info(f"First half {split_factor:.0%} slower, second half {split_factor:.0%} faster")
            else:  # Positive Split
                split_factor = st.slider("First Half Adjustment:", min_value=0.01, max_value=0.10, value=0.05, step=0.01, format="-%g")
                st.info(f"First half {split_factor:.0%} faster, second half {split_factor:.0%} slower")
                split_factor = -split_factor  # Negate for calculations
        
        with col2:
            st.subheader("Pace Input")
            
            # Option to input by pace or time
            input_method = st.radio(
                "Input Method",
                ["Target Pace", "Target Time"],
                horizontal=True
            )
            
            if input_method == "Target Pace":
                pace_unit = st.radio("Pace Unit", ["min/km", "min/mile"], horizontal=True)
                
                if pace_unit == "min/km":
                    pace_input = st.text_input("Target Pace (min:sec per km)", "5:00")
                    try:
                        pace_parts = pace_input.split(":")
                        if len(pace_parts) == 2:
                            pace_min, pace_sec = int(pace_parts[0]), int(pace_parts[1])
                            pace_seconds_per_km = pace_min * 60 + pace_sec
                        else:
                            pace_seconds_per_km = float(pace_input) * 60
                        
                        # Calculate total time
                        total_seconds = pace_seconds_per_km * race_distance_km
                        
                    except ValueError:
                        st.error("Please enter a valid pace in the format min:sec or decimal minutes")
                        pace_seconds_per_km = 0
                        total_seconds = 0
                else:
                    pace_input = st.text_input("Target Pace (min:sec per mile)", "8:00")
                    try:
                        pace_parts = pace_input.split(":")
                        if len(pace_parts) == 2:
                            pace_min, pace_sec = int(pace_parts[0]), int(pace_parts[1])
                            pace_seconds_per_mile = pace_min * 60 + pace_sec
                        else:
                            pace_seconds_per_mile = float(pace_input) * 60
                        
                        # Convert to seconds per km
                        pace_seconds_per_km = pace_seconds_per_mile / 1.60934
                        
                        # Calculate total time
                        total_seconds = pace_seconds_per_km * race_distance_km
                        
                    except ValueError:
                        st.error("Please enter a valid pace in the format min:sec or decimal minutes")
                        pace_seconds_per_km = 0
                        total_seconds = 0
            else:
                # Input target finish time
                col_h, col_m, col_s = st.columns(3)
                with col_h:
                    hours = st.number_input("Hours", min_value=0, max_value=24, value=0)
                with col_m:
                    minutes = st.number_input("Minutes", min_value=0, max_value=59, value=20)
                with col_s:
                    seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0)
                
                total_seconds = hours * 3600 + minutes * 60 + seconds

                # Calculate pace
                if race_distance_km > 0:
                    pace_seconds_per_km = total_seconds / race_distance_km
                else:
                    pace_seconds_per_km = 0
            
            # Calculate and display results if valid pace
            if pace_seconds_per_km > 0:
                # Convert back to min:sec format
                pace_min = int(pace_seconds_per_km // 60)
                pace_sec = int(pace_seconds_per_km % 60)
                pace_km_formatted = f"{pace_min}:{pace_sec:02d}"
                
                # Calculate mi/pace
                pace_seconds_per_mile = pace_seconds_per_km * 1.60934
                pace_mile_min = int(pace_seconds_per_mile // 60)
                pace_mile_sec = int(pace_seconds_per_mile % 60)
                pace_mile_formatted = f"{pace_mile_min}:{pace_mile_sec:02d}"
                
                # Calculate finish time
                finish_time = seconds_to_time_str(total_seconds)
                
                st.markdown(f"""
                <div class="result-box" style="margin-top: 20px; text-align: center;">
                    <h3 style="color: #E6754E;">Race Summary</h3>
                    <div style="font-size: 24px; font-weight: bold; margin: 15px 0;">
                        {finish_time}
                    </div>
                    <div style="font-size: 16px; margin-bottom: 5px;">
                        Pace: {pace_km_formatted} min/km | {pace_mile_formatted} min/mile
                    </div>
                    <div style="font-size: 14px; color: #666;">
                        Speed: {(60 / (pace_seconds_per_km / 60)):.2f} km/h | {(60 / (pace_seconds_per_km / 60) / 1.60934):.2f} mph
                    </div>
                </div>
                """, unsafe_allow_html=True)
