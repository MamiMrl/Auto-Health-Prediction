import streamlit as st
import numpy as np
import pandas as pd
import joblib

#---------------------------------#

st.set_page_config(page_title='Auto Health Prediction App', page_icon=':car:', 
                menu_items={
                    'Get Help': 'https://www.linkedin.com/in/muhammedmaral/',
                    'About': "For More Information\n" + "https://github.com/MamiMrl"
                }, layout='centered', initial_sidebar_state='auto')

# Custom CSS for Styling
st.markdown("""
<style>
.bold-red {
    color: #BB0000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

#---------------------------------#

# Title and Introduction
st.title("Auto Health Prediction")
st.markdown("""
Welcome to the <span class="bold-red">Auto Health Prediction</span> platform. This system integrates with your car's sensors to anticipate engine issues, enhancing safety and reliability.
""", unsafe_allow_html=True)


st.image('img/unsplash_engine.jpg', width=800)

st.header("Data Dictionary")
st.markdown("""
- <span class="bold-red">Lub_oil_pressure</span>: Ensures smooth engine operation by measuring the lubricating oil pressure.
- <span class="bold-red">Coolant_temp</span>: Reflects engine heat efficiency through coolant temperature.
- <span class="bold-red">Engine_condition</span>: Binary indicator displaying the health status of the engine.
- <span class="bold-red">log_Engine_RPM</span>: Provides a normalized view of engine speed in logarithmic scale.
- <span class="bold-red">log_Fuel_pressure</span>: Log-transformed metric offering insights into the fuel system's performance.
- <span class="bold-red">log_Coolant_pressure</span>: Helps in detecting cooling system anomalies through a logarithmic pressure value.
- <span class="bold-red">log_Lub_oil_temp</span>: Essential for assessing oil temperature and engine protection in logarithmic form.
- <span class="bold-red">pressure_ratio</span>: Composite metric for engine health evaluation based on various pressure readings.
- <span class="bold-red">temp_diff</span>: Diagnoses cooling performance by analyzing temperature differences.
- <span class="bold-red">rpm_pressure_ratio</span>: Compound metric for engine analysis combining RPM and pressure data.
- <span class="bold-red">normalized_fuel_pressure</span>: Crucial for understanding fuel delivery efficiency, standardized for comparison.
- <span class="bold-red">normalized_temp</span>: Provides a consistent scale for temperature data comparison across engines.
- <span class="bold-red">max_coolant_temp</span>: Maximum coolant temperature recorded, key in identifying thermal issues.
- <span class="bold-red">min_coolant_pressure</span>: Minimum coolant pressure recorded, vital for leak detection.
- <span class="bold-red">high_engine_rpm</span>: Indicator of engine operating at high RPMs, a sign of potential overuse.
""", unsafe_allow_html=True)

#---------------------------------#

st.markdown("---")

#---------------------------------#

st.header("DataFrame Overview")
st.markdown("Swipe Right to View More 👉")
df = pd.read_csv('auto_health_dataset.csv')

st.table(df.sample(10, random_state=42))

#---------------------------------#

# How It Works Section
st.header("How It Works")
st.markdown("""
1. <span class="bold-red">Data Collection</span>: Gathering real-time data from your vehicle's sensors.
2. <span class="bold-red">Analysis</span>: Utilizing machine learning to analyze engine performance.
3. <span class="bold-red">Prediction</span>: Assessing the likelihood of future engine defects.
""", unsafe_allow_html=True)

#---------------------------------#

st.image('img/car_image_generated.png', width=800)

#---------------------------------#

# Benefits Section
st.header("Benefits")
st.markdown("""
- <span class="bold-red">Proactive Maintenance</span>: Address issues before they escalate.
- <span class="bold-red">Safety</span>: Keep informed about your vehicle's health.
- <span class="bold-red">Peace of Mind</span>: Drive confidently with continuous monitoring.
""", unsafe_allow_html=True)

#---------------------------------#

# Integration Information
st.header("Vehicle Integration")
st.markdown("""
Connect this platform with your vehicle's sensor system for a seamless experience. Visit our [integration guide](#) for instructions.
""", unsafe_allow_html=True)

#---------------------------------#

# Contact and Support
st.header("Support")
st.markdown("""
Need help or have questions? Feel free to [contact us](#).
""", unsafe_allow_html=True)

#---------------------------------#

# Footer
st.markdown("---")
st.markdown("""
Auto Health Prediction - <span class="bold-red">Predicting the Unpredictable</span> for a smoother journey 🚗
""", unsafe_allow_html=True)

#---------------------------------#

st.sidebar.header("Car Sensory Input Parameters")

car_id = st.sidebar.number_input("Car ID", min_value=1, format="%d")
owner_name = st.sidebar.text_input("Owner Name")
owner_surname = st.sidebar.text_input("Owner Surname")
owner_contact = st.sidebar.text_input("Owner Contact Info (phone number)")

lub_oil_pressure = st.sidebar.slider("Lubricating Oil Pressure", min_value=0.0, max_value=8.0, step=0.1)
coolant_temp = st.sidebar.slider("Coolant Temperature", min_value=60.0, max_value=100.0, step=0.1)
log_engine_rpm = st.sidebar.slider("Log Engine RPM", min_value=5.5, max_value=8.0, step=0.1)
log_fuel_pressure = st.sidebar.slider("Log Fuel Pressure", min_value=0.0, max_value=3.5, step=0.1)
log_coolant_pressure = st.sidebar.slider("Log Coolant Pressure", min_value=0.0, max_value=2.5, step=0.1)
log_lub_oil_temp = st.sidebar.slider("Log Lubricating Oil Temperature", min_value=4.2, max_value=4.6, step=0.1)
pressure_ratio = st.sidebar.number_input("Pressure Ratio", min_value=0.0, max_value=600.0, step=1.0)
temp_diff = st.sidebar.slider("Temperature Difference", min_value=-25.0, max_value=25.0, step=0.1)
rpm_pressure_ratio = st.sidebar.number_input("RPM Pressure Ratio", min_value=0, max_value=130000, step=1000)
normalized_fuel_pressure = st.sidebar.slider("Normalized Fuel Pressure", min_value=0.0, max_value=3.5, step=0.1)
normalized_temp = st.sidebar.slider("Normalized Temperature", min_value=0.7, max_value=1.3, step=0.01)
max_coolant_temp = st.sidebar.slider("Max Coolant Temperature", min_value=75.0, max_value=100.0, step=0.1)
min_coolant_pressure = st.sidebar.slider("Min Coolant Pressure", min_value=0.0, max_value=3.0, step=0.1)
high_engine_rpm = st.sidebar.selectbox("High Engine RPM", [0, 1])  # Assuming this is a binary choice

#---------------------------------#

# Prediction
model = joblib.load('random_forest_model.pkl')

# Button to make prediction
if st.sidebar.button('Predict Engine Condition'):
    # Create a DataFrame to make the prediction
    input_data = pd.DataFrame({
        'Lub_oil_pressure': [lub_oil_pressure],
        'Coolant_temp': [coolant_temp],
        'log_Engine_RPM': [log_engine_rpm],
        'log_Fuel_pressure': [log_fuel_pressure],
        'log_Coolant_pressure': [log_coolant_pressure],
        'log_Lub_oil_temp': [log_lub_oil_temp],
        'pressure_ratio': [pressure_ratio],
        'temp_diff': [temp_diff],
        'rpm_pressure_ratio': [rpm_pressure_ratio],
        'normalized_fuel_pressure': [normalized_fuel_pressure],
        'normalized_temp': [normalized_temp],
        'max_coolant_temp': [max_coolant_temp],
        'min_coolant_pressure': [min_coolant_pressure],
        'high_engine_rpm': [high_engine_rpm]
    })

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction
    condition = 'Good' if prediction[0] == 1 else 'Bad'
    st.write(f"The predicted engine condition is: **{condition}**")

# Modify your prediction code to output probabilities
    prediction_prob = np.round(model.predict_proba(input_data.values), 3)
    st.write(f"Probability of being Good: {prediction_prob[0][1]:.2f}")
    st.write(f"Probability of being Bad: {prediction_prob[0][0]:.2f}")

    # Log input data
    print(input_data)



    # Optionally, display more information about the prediction
    # For example, probabilities or other model outputs