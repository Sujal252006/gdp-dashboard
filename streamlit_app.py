
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Gym Predictive Analytics",
    page_icon="🏋️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("gym_calorie_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(
        "Model not found. Please upload "
        "gym_calorie_model.pkl into the same folder as app.py."
    )
    st.stop()

# Dashboard heading
st.title("🏋️ Gym Predictive Analytics")
st.subheader("Calories Burned Prediction Using Machine Learning")

st.write(
    "Enter your workout details below to estimate "
    "the calories burned during your gym session."
)

st.divider()

# Input form
with st.form("workout_form"):

    st.header("Enter Workout Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=10, max_value=100, value=25
        )

        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        weight = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=300.0,
            value=70.0, step=1.0
        )

        height = st.number_input(
            "Height (m)", min_value=1.0, max_value=2.5,
            value=1.70, step=0.01
        )

        max_bpm = st.number_input(
            "Maximum BPM", min_value=60, max_value=220, value=180
        )

        avg_bpm = st.number_input(
            "Average BPM", min_value=40, max_value=200, value=130
        )

        resting_bpm = st.number_input(
            "Resting BPM", min_value=30, max_value=120, value=70
        )

    with col2:
        duration = st.number_input(
            "Session Duration (hours)",
            min_value=0.1, max_value=8.0,
            value=1.0, step=0.1
        )

        workout_type = st.selectbox(
            "Workout Type",
            ["Cardio", "HIIT", "Strength", "Yoga"]
        )

        fat_percentage = st.number_input(
            "Fat Percentage (%)",
            min_value=1.0, max_value=70.0,
            value=25.0, step=0.5
        )

        water_intake = st.number_input(
            "Water Intake (liters)",
            min_value=0.0, max_value=15.0,
            value=2.5, step=0.1
        )

        workout_frequency = st.number_input(
            "Workout Frequency (days/week)",
            min_value=1, max_value=7, value=4
        )

        experience_level = st.selectbox(
            "Experience Level", [1, 2, 3],
            format_func=lambda x: {
                1: "Beginner",
                2: "Intermediate",
                3: "Advanced"
            }[x]
        )

    submitted = st.form_submit_button(
        "Predict Calories Burned", type="primary",
        use_container_width=True
    )

# Prediction
if submitted:
    if avg_bpm > max_bpm:
        st.error("Average BPM cannot be greater than Maximum BPM.")
    elif resting_bpm > avg_bpm:
        st.error("Resting BPM should not exceed Average BPM.")
    else:
        bmi = weight / (height ** 2)

        # Match the feature names used during model training
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Weight (kg)": weight,
            "Height (m)": height,
            "Max_BPM": max_bpm,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration (hours)": duration,
            "Workout_Type": workout_type,
            "Fat_Percentage": fat_percentage,
            "Water_Intake (liters)": water_intake,
            "Workout_Frequency (days/week)": workout_frequency,
            "Experience_Level": experience_level,
            "BMI": bmi
        }])

        try:
            prediction = model.predict(input_data)[0]

            st.divider()
            st.header("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    label="Estimated Calories Burned",
                    value=f"{prediction:.2f} kcal"
                )

            with result_col2:
                st.metric(
                    label="Calculated BMI",
                    value=f"{bmi:.2f}"
                )

            st.success("Prediction completed successfully!")

            st.caption(
                "Note: This is an estimate from a machine-learning model, "
                "not a direct measurement of calories burned."
            )

            with st.expander("View Input Data"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Check that this model was trained with the same "
                "14 input features and compatible category values."
            )

st.divider()
st.caption("Gym Predictive Analytics | Random Forest Regression")
