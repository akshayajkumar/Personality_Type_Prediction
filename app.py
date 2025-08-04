import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('personality_prediction.pkl', 'rb') as file:
    data = pickle.load(file)

st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Personality Type Predictor")

st.markdown(
    "<h4 style='text-align: center;'>Welcome! Fill in the details below to predict your personality type.</h4>",
    unsafe_allow_html=True
)
# 👇 Add the image below the title
st.image("https://cdn.vectorstock.com/i/500p/19/59/introvert-and-extrovert-infographic-set-vector-42441959.jpg", use_container_width=True)




# Input section
st.subheader("📋 Input Your Details")

col1, col2 = st.columns(2)

with col1:
    time_alone = st.number_input('🕒 Time Spent Alone (hours/day)', min_value=0.0, max_value=24.0, step=1.0)
    stage_fear = st.radio('🎤 Do you have stage fear?', ['Yes', 'No'])
    social_events = st.slider('🎉 Social Event Attendance (per month)', 0, 30, 5)

with col2:
    going_outside = st.slider('🚶 How often do you go outside (per week)?', 0, 14, 4)
    drained_after_social = st.radio('😫 Feel Drained After Socializing?', ['Yes', 'No'])
    friends_circle_size = st.slider('👥 Number of Close Friends', 0, 50, 5)
    post_frequency = st.slider('📱 Social Media Post Frequency (per week)', 0, 30, 2)

# Add vertical spacing before the button
st.markdown("###")

# Create 3 columns and center the button in the middle one
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict = st.button("🔍 Predict Personality")



# Processing prediction
if predict:
    # Encode categorical inputs
    stage_fear_encoded = 1 if stage_fear == 'Yes' else 0
    drained_encoded = 1 if drained_after_social == 'Yes' else 0

    test_input = [[
        time_alone,
        stage_fear_encoded,
        social_events,
        going_outside,
        drained_encoded,
        friends_circle_size,
        post_frequency
    ]]

    if data.get("scaled"):
        test_input = data["scaled"].transform(test_input)

    prediction = data["model"].predict(test_input)[0]
    label_map = {0: "Extrovert", 1: "Introvert"}
    personality = label_map.get(prediction, "Unknown")

    st.success(f"🎯 Your Predicted Personality Type is: **{personality}**")
