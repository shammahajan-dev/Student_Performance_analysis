# Import Streamlit
# Streamlit is used to create our web application
import streamlit as st

# Import pickle
# Pickle is used to load the trained machine learning model
import pickle

# Import NumPy
# NumPy is used to create the input data in array format
import numpy as np


# --------------------------------------------------
# STEP 1: Load the trained model
# --------------------------------------------------

# Path of the saved model
model_path = "model.pkl"

# Open the model file
with open(model_path, "rb") as file:

    # Load the trained model
    model = pickle.load(file)


# --------------------------------------------------
# STEP 2: Create the Streamlit application
# --------------------------------------------------

# Display the title of the application
st.title("Student Placement Prediction")

# Display a short description
st.write("Enter student details to predict placement.")


# --------------------------------------------------
# STEP 3: Take input from the user
# --------------------------------------------------

# Create input boxes for the features

feature1 = st.number_input("Feature 1", value=0)

feature2 = st.number_input("Feature 2", value=0)

feature3 = st.number_input("Feature 3", value=0)

feature4 = st.number_input("Feature 4", value=0)


# --------------------------------------------------
# STEP 4: Create Predict button
# --------------------------------------------------

# This block will execute when the user clicks
# the Predict button

if st.button("Predict"):

    # Store all user inputs in a list
    int_features = [
        feature1,
        feature2,
        feature3,
        feature4
    ]

    # Convert the list into a NumPy array
    # ML model expects data in array format
    final_features = np.array([int_features])


    # --------------------------------------------------
    # STEP 5: Make prediction
    # --------------------------------------------------

    # Send the input data to the trained model
    prediction = model.predict(final_features)


    # --------------------------------------------------
    # STEP 6: Display the result
    # --------------------------------------------------

    # If prediction is 1, student is placed
    if prediction[0] == 1:

        st.success("Prediction: Placed")

    # Otherwise, student is not placed
    else:

        st.error("Prediction: Not Placed")