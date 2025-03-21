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

# Import the logo setup function - if you have it
try:
    from logo_setup import create_logo
except ImportError:
    # Define a minimal logo creation function if the import fails
    def create_logo():
        return None

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

# Create and setup logo
try:
    logo_path = create_logo()
    if logo_path and os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
    else:
        # Create a very simple text logo if image is not available
        st.sidebar.title("RUNNER")
        st.sidebar.title("CALCULATOR")
except Exception as e:
    st.sidebar.title("RUNNER")
    st.sidebar.title("CALCULATOR")

st.sidebar.markdown("---")
st.sidebar.title("Runner Performance Calculator")
st.sidebar.markdown("Predict race times and training zones based on your previous performances")

# Now directly import the runner calculator code
from runner_calculator import main as run_calculator

# Run the calculator function
run_calculator()
