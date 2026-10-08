# Import Streamlit
# Streamlit is used to create our web application
import streamlit as st

# Import pickle
# Pickle is used to load the trained machine learning model
import pickle

# Import NumPy
# NumPy is used to create input data in array format
import numpy as np


# --------------------------------------------------
# STEP 1: Load the trained model
# --------------------------------------------------

# Path of the saved model
model_path = "Model.pkl"

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

# Take IQ value from the user
IQ = st.number_input(
    "IQ",
    min_value=0.0,
    value=70.0
)

# Take CGPA value from the user
CGPA = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

# Take 10th marks from the user
Marks_10th = st.number_input(
    "10th Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

# Take 12th marks from the user
Marks_12th = st.number_input(
    "12th Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

# Take Communication Skills value from the user
Communication_Skills = st.number_input(
    "Communication Skills",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)


# --------------------------------------------------
# STEP 4: Create Predict button
# --------------------------------------------------

# This block will execute when the user clicks
# the Predict button

if st.button("Predict"):

    # Store all five input values in a list
    # Keep the same order used during model training
    input_data = [
        IQ,
        CGPA,
        Marks_10th,
        Marks_12th,
        Communication_Skills
    ]


    # --------------------------------------------------
    # STEP 5: Convert input into NumPy array
    # --------------------------------------------------

    # Convert the list into a NumPy array
    # The machine learning model expects data
    # in 2D array format
    final_features = np.array([input_data])


    # --------------------------------------------------
    # STEP 6: Make prediction
    # --------------------------------------------------

    # Send the input data to the trained model
    prediction = model.predict(final_features)


    # --------------------------------------------------
    # STEP 7: Display the result
    # --------------------------------------------------

    # If prediction is 1, student is placed
    if prediction[0] == 1:

        st.success("Prediction: Placed")

    # If prediction is 0, student is not placed
    else:

        st.error("Prediction: Not Placed")