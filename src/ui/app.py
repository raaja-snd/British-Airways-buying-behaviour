import sys
import streamlit as st
import pandas as pd
import joblib
import yaml
from pathlib import Path

# --- CONFIG & MODEL LOADING ---
@st.cache_resource
def load_assets():
    parent_path = Path(__file__).resolve().parents[2]
    if str(parent_path) is not sys.path:
        sys.path.insert(0,str(parent_path))
    yaml_path = parent_path /'src/config.yaml'
    
    with open(yaml_path) as c:
        config = yaml.safe_load(c)
    
    # Update this path logic to match your local setup
    model_path = parent_path / config['model']['cat_boost']['pipeline']
    pipeline = joblib.load(model_path)
    return pipeline

pipeline = load_assets()

# --- BRANDING & STYLING ---
st.set_page_config(page_title="BA Booking Predictor", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #075aaa; color: white; border-radius: 5px; width: 100%; }
    .stHeader { color: #075aaa; }
    </style>
    """, unsafe_allow_html=True)

st.title("✈️ British Airways Booking Prediction")
st.markdown("Predict the likelihood of a customer completing a flight booking.")

# --- TABS FOR PREDICTION MODES ---
tab1, tab2 = st.tabs(["Single Customer Prediction", "Batch CSV Prediction"])

with tab1:
    st.header("Customer Details")
    
    # Organizing inputs into columns for a better design
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
        sales_channel = st.selectbox("Sales Channel", ["Internet", "Mobile"])
        trip_type = st.selectbox("Trip Type", ["Round Trip", "One Way", "Circle Trip"])
        purchase_lead = st.number_input("Purchase Lead (Days)", min_value=0, value=10)

    with col2:
        length_of_stay = st.number_input("Length of Stay (Days)", min_value=0, value=7)
        flight_hour = st.selectbox("Flight Hour", list(range(24)))
        flight_day = st.selectbox("Flight Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        flight_duration = st.number_input("Flight Duration (Hours)", min_value=0.0, value=5.5)

    with col3:
        route = st.text_input("Route (e.g., AKLHND)", value="AKLHND")
        booking_origin = st.text_input("Booking Origin (Country)", value="Australia")
        st.markdown("**Extra Services**")
        wants_extra_baggage = st.checkbox("Extra Baggage")
        wants_preferred_seat = st.checkbox("Preferred Seat")
        wants_in_flight_meals = st.checkbox("In-flight Meals")

    # Mapping inputs to a DataFrame
    input_data = pd.DataFrame([{
        'num_passengers': num_passengers,
        'sales_channel': sales_channel,
        'trip_type': trip_type,
        'purchase_lead': purchase_lead,
        'length_of_stay': length_of_stay,
        'flight_hour': flight_hour,
        'flight_day': flight_day,
        'route': route,
        'booking_origin': booking_origin,
        'wants_extra_baggage': 1 if wants_extra_baggage else 0,
        'wants_preferred_seat': 1 if wants_preferred_seat else 0,
        'wants_in_flight_meals': 1 if wants_in_flight_meals else 0,
        'flight_duration': flight_duration
    }])

    if st.button("Predict Booking Completion"):
        # The pipeline handles FeatureEngineering and Preprocessing internally!
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"✅ High Likelihood of Booking! (Probability: {probability:.2%})")
        else:
            st.error(f"❌ Low Likelihood of Booking. (Probability: {probability:.2%})")

with tab2:
    st.header("Batch Processing")
    st.write("Upload a CSV file with the same columns as the original dataset (excluding the target).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        if st.button("Process Batch"):
            with st.spinner('Running predictions...'):
                # Handle predictions
                results = pipeline.predict(batch_df)
                probs = pipeline.predict_proba(batch_df)[:, 1]
                
                batch_df['Prediction'] = results
                batch_df['Booking_Probability'] = probs
                
                st.write(batch_df)
                
                # Download button for results
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="ba_predictions.csv",
                    mime="text/csv",
                )