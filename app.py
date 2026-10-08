# ==============================================================
# PlacePredict - Student Placement Prediction (Streamlit version)
# Run this file with:  streamlit run app.py
# ==============================================================

import pickle                 # Used to load the saved ML model
import numpy as np            # Used to create the input in array format
import streamlit as st        # Streamlit builds the web page for us

# --------------------------------------------------------------
# 1. PAGE SETTINGS
# This must be the FIRST Streamlit command in the file.
# --------------------------------------------------------------
st.set_page_config(
    page_title="PlacePredict | Student Placement Prediction",
    page_icon="🎓",
    layout="centered",
)

# --------------------------------------------------------------
# 2. LOAD THE MODEL
# The model file "Model.pkl" must be in the same folder as app.py.
# @st.cache_resource loads the model only once, so the app stays fast.
# --------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("Model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# --------------------------------------------------------------
# 3. CUSTOM CSS
# Streamlit allows CSS inside st.markdown (with unsafe_allow_html=True).
# JavaScript is NOT allowed here, but we do not need it,
# because Streamlit sliders already work by themselves.
# --------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Load the Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Use the font on the whole app and set a light background */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: #f1f5f9;
        }

        /* Hide the default Streamlit menu, header and footer for a clean look */
        #MainMenu, header, footer { visibility: hidden; }

        /* Keep the content width neat and reduce the top gap */
        .block-container {
            max-width: 760px;
            padding-top: 1.5rem;
        }

        /* ---------- HERO SECTION (top dark banner) ---------- */
        .hero {
            background:
                radial-gradient(circle at 85% 15%, rgba(99,102,241,0.5), transparent 45%),
                radial-gradient(circle at 10% 90%, rgba(34,211,238,0.25), transparent 40%),
                #0f172a;
            color: #fff;
            text-align: center;
            padding: 44px 24px 40px;
            border-radius: 22px;
            margin-bottom: 22px;
        }
        .hero .brand {
            display: inline-block;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            color: #22d3ee;
            border: 1px solid rgba(34,211,238,0.4);
            padding: 5px 14px;
            border-radius: 999px;
            margin-bottom: 16px;
        }
        .hero h1 {
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -1px;
            line-height: 1.15;
            color: #fff;
            margin: 0;
            padding: 0;
        }
        /* Gradient colored word */
        .hero h1 span {
            background: linear-gradient(90deg, #818cf8, #22d3ee);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .hero p {
            color: #94a3b8;
            font-size: 15px;
            max-width: 480px;
            margin: 14px auto 0;
        }

        /* ---------- FORM CARD ---------- */
        /* st.form creates this container, so we style it as a white card */
        [data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.10);
        }

        /* Labels of sliders and number boxes */
        [data-testid="stWidgetLabel"] p {
            font-weight: 600;
            font-size: 14px;
            color: #0f172a;
        }

        /* ---------- SUBMIT BUTTON ---------- */
        .stFormSubmitButton button {
            width: 100%;
            padding: 12px 0;
            font-size: 16px;
            font-weight: 700;
            color: #fff;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            border: none;
            border-radius: 12px;
            box-shadow: 0 10px 22px rgba(79, 70, 229, 0.35);
            transition: all 0.2s ease;
        }
        .stFormSubmitButton button:hover {
            transform: translateY(-2px);          /* Lift up on hover */
            box-shadow: 0 14px 28px rgba(79, 70, 229, 0.45);
            color: #fff;
        }

        /* ---------- RESULT CARD ---------- */
        .result {
            text-align: center;
            padding: 28px 20px;
            border-radius: 20px;
            margin-top: 22px;
            animation: pop 0.5s ease;             /* Small pop-in effect */
        }
        .result.good { background:#ecfdf5; border:2px solid #6ee7b7; color:#059669; }
        .result.bad  { background:#fef2f2; border:2px solid #fca5a5; color:#dc2626; }
        .result .icon  { font-size: 46px; }
        .result .label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            opacity: 0.8;
            margin-top: 6px;
        }
        .result .text  { font-size: 24px; font-weight: 800; margin-top: 4px; }
        @keyframes pop {
            from { opacity: 0; transform: scale(0.92); }
            to   { opacity: 1; transform: scale(1); }
        }

        /* ---------- NOTE AT THE BOTTOM ---------- */
        .note {
            text-align: center;
            color: #64748b;
            font-size: 13px;
            margin-top: 22px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# 4. HERO SECTION (HTML + CSS only, so it works in st.markdown)
# --------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="brand">🎓 PLACEPREDICT · ML POWERED</div>
        <h1>Will you get <span>placed?</span></h1>
        <p>Enter your academic and skill details. Our machine learning model
        studies the pattern and predicts your placement chances.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# 5. THE FORM
# st.form groups all inputs, so the page runs the prediction
# only when the user clicks the button (not on every slider move).
# --------------------------------------------------------------
with st.form("predict_form"):
    st.subheader("Your Profile")
    st.caption("Move the sliders to enter your details.")

    # Two columns: IQ and CGPA on one row
    col1, col2 = st.columns(2)
    with col1:
        iq = st.slider("🧠 IQ", min_value=70, max_value=160, value=100)
    with col2:
        cgpa = st.slider("📊 CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.01)

    # Next row: 10th and 12th marks
    col3, col4 = st.columns(2)
    with col3:
        marks_10 = st.slider("📘 10th Marks", min_value=0, max_value=100, value=70)
    with col4:
        marks_12 = st.slider("📗 12th Marks", min_value=0, max_value=100, value=70)

    # Last input: Communication skills (full width)
    comm = st.slider("🗣️ Communication Skills", min_value=0.0, max_value=10.0, value=5.0, step=0.01)

    # The submit button
    submitted = st.form_submit_button("🚀 Predict My Placement")

# --------------------------------------------------------------
# 6. PREDICTION AND RESULT
# This part runs only after the button is clicked.
# --------------------------------------------------------------
if submitted:
    # Keep the same order that was used while training the model:
    # IQ, CGPA, 10th Marks, 12th Marks, Communication Skills
    input_data = [iq, cgpa, marks_10, marks_12, comm]

    # The model expects a 2D array, so we put the list inside another list
    final_features = np.array([input_data])

    # The model returns 1 (Placed) or 0 (Not Placed)
    prediction = model.predict(final_features)[0]

    if prediction == 1:
        st.markdown(
            """
            <div class="result good">
                <div class="icon">🎉</div>
                <div class="label">Prediction Result</div>
                <div class="text">Prediction: Placed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result bad">
                <div class="icon">📉</div>
                <div class="label">Prediction Result</div>
                <div class="text">Prediction: Not Placed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------
# 7. SMALL NOTE
# --------------------------------------------------------------
st.markdown(
    '<div class="note">💡 This is a prediction from a model, not a guarantee. '
    'Use it as guidance and keep improving your skills.</div>',
    unsafe_allow_html=True,
)