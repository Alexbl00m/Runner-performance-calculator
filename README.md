# Runner Performance Calculator

A comprehensive web application for runners to calculate race predictions, training zones, and pace strategies.

## Features

- **Race Time Predictor**: Predict finish times for various race distances based on your previous performances
- **Training Zones Calculator**: Get personalized Yousli 7-zone training paces based on your threshold
- **Race Pace Calculator**: Plan your race strategy with detailed lap breakdowns and split times
- **Visual Analytics**: Interactive charts and visualizations for better race planning

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/runner-performance-calculator.git
   cd runner-performance-calculator
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Race Predictor
Input up to three of your best recent race performances, and the calculator will:
- Generate personalized predictions for common race distances
- Calculate your unique fatigue factor based on your provided data
- Show predictions in both time and pace formats
- Include a triathlon brick run adjustment for multisport athletes

### Training Zones
View your optimal training zones based on the Yousli 7-zone system:
- Each zone includes specific pace ranges in both min/km and min/mile
- Detailed descriptions for each training zone help you understand their purpose
- Adapt your training approach based on these scientifically-based zones

### Race Pace Calculator
Plan your race execution with:
- Even, negative, or positive split strategies
- Detailed kilometer or mile splits
- Visualization of your pacing plan
- Total race time calculation

## Science Behind the Calculator

The Runner Performance Calculator uses advanced mathematical models:

1. **Race Prediction** - Uses a modified version of Peter Riegel's formula with an adaptive fatigue factor:
   ```
   T2 = T1 × (D2/D1)^f
   ```
   Where:
   - T2 is the predicted time for distance D2
   - T1 is your known time for distance D1
   - f is the fatigue factor (typically between 1.05 and 1.15)

2. **Training Zones** - The Yousli 7-zone system is based on your threshold pace, with zones calculated as percentages of this threshold.

## Customization

- Place your logo in the root directory as "logo.png" to use your custom branding
- Edit the CSS in app.py to match your brand colors

## License

This project is licensed under the MIT License - see the LICENSE file for details.
