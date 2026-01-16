import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
clf = pickle.load(open('model (1).pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

# Title
st.title("Placement Prediction App")
st.write("Enter student details to predict placement status")

# Inputs
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ", min_value=0.0, max_value=300.0, step=1.0)

# Predict button
if st.button("Predict Placement"):
    # Convert input into 2D numpy array
    input_data = np.array([[cgpa, iq]])

    # Scale using trained scaler
    scaled_input = scaler.transform(input_data)

    # Predict using trained model
    prediction = clf.predict(scaled_input)

    # Output
    if prediction[0] == 1:
        st.success("✅ Person is likely to be PLACED!")
    else:
        st.error("❌ Person is NOT likely to be placed.")

# Footer
st.markdown("---")
st.markdown("Made by Snehasish Porel 🚀")

