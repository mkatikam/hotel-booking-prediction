import streamlit as st
import pandas as pd
import numpy as np
import dill
import gdown
import mlflow
import mlflow.sklearn
from preprocess import data_preprocess, make_prediction

# Define feature variables
features_var = ['lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number',
                'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
                'adults', 'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                'booking_changes', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
                'total_of_special_requests']

# Sidebar inputs
st.sidebar.header("Input Features")
lead_time = st.sidebar.slider("Lead Time", min_value=0, max_value=1000, value=0)
arrival_year = st.sidebar.slider("Arrival Date Year", min_value=2000, max_value=2100, value=2023)
arrival_month = st.sidebar.slider("Arrival Date Month", min_value=1, max_value=12, value=1)
arrival_date = st.sidebar.slider("Arrival Date Week Number", min_value=1, max_value=53, value=1)
no_of_weekend_nights = st.sidebar.slider("Stays in Weekend Nights", min_value=0, max_value=10, value=0)
no_of_week_nights = st.sidebar.slider("Stays in Week Nights", min_value=0, max_value=30, value=0)
no_of_adults = st.sidebar.slider("Adults", min_value=1, max_value=10, value=1)
no_of_children = st.sidebar.slider("Children", min_value=0, max_value=10, value=0)
type_of_meal_plan = st.sidebar.slider("Meal Type", min_value=1, max_value=4, value=1)
babies = st.sidebar.slider("Babies", min_value=0, max_value=10, value=0)
no_of_previous_cancellations = st.sidebar.slider("Previous Cancellations", min_value=0, max_value=100, value=0)
no_of_previous_bookings_not_canceled = st.sidebar.slider("Previous Bookings Not Canceled", min_value=0, max_value=100, value=0)
booking_changes = st.sidebar.slider("Booking Changes", min_value=0, max_value=100, value=0)
days_in_waiting_list = st.sidebar.slider("Days in Waiting List", min_value=0, max_value=1000, value=0)
required_car_parking_space = st.sidebar.slider("Required Car Parking Spaces", min_value=0, max_value=10, value=0)
total_of_special_requests = st.sidebar.slider("Total of Special Requests", min_value=0, max_value=10, value=0)
room_type_reserved = st.sidebar.slider("Room Type", min_value=1, max_value=4, value=1)
market_segment_type = st.sidebar.selectbox("Market Segment Type", ["Online", "Offline", "Complementary", "Corporate", "Aviation"])
avg_price_per_room = st.sidebar.slider("Average Avg Price Per Room", min_value=0, max_value=500, value=1 )
no_of_special_requests = st.sidebar.slider("No of Special Requests", min_value=1, max_value=3, value=1 )

def get_model():
    url = 'https://drive.google.com/file/d/1Pvm-XaVQN46XkTYk4cqY7_VZXEkdyCQv/view?usp=sharing'
    output = 'downloaded_rfr_v1.pkl'
    gdown.download(url, output, quiet=False, fuzzy=True)
    with open('downloaded_rfr_v1.pkl', 'rb') as f:
        reloaded_model = dill.load(f)
    return reloaded_model

def get_model_local():
    with open('./app/rfg_model.pkl','rb') as f:
        reloaded_modal = dill.load(f)
    return reloaded_modal

st.title('Hotel Booking Prediction')

def format_output(data):
    output_res = []
    for i in data:
        if i==0:
            output_res.append("Cancelled")
        else:
            output_res.append("Not Cancelled")
    return output_res

def input_params():
    return ({
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'required_car_parking_space': required_car_parking_space,
        'no_of_special_requests': total_of_special_requests,
        'avg_price_per_room': avg_price_per_room,
        'repeated_guest': 0,
        'market_segment_type': market_segment_type,
        'type_of_meal_plan': type_of_meal_plan,
        'room_type_reserved': room_type_reserved
    })

st.write(input_params())

if st.button('Predict'):
    input_df = pd.DataFrame({
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'required_car_parking_space': [required_car_parking_space],
        'no_of_special_requests': [total_of_special_requests],
        'avg_price_per_room': [avg_price_per_room],
        'repeated_guest': 0,
        'market_segment_type': [market_segment_type],
        'type_of_meal_plan': [type_of_meal_plan],
        'room_type_reserved': [room_type_reserved]
    })
    
    df = pd.read_csv('train_data_mapping.csv')

    st.write(input_df)
    pred_model = get_model_local()

    X, y = data_preprocess(df)
    
    with mlflow.start_run():
        y_hat, y_hat_overall = make_prediction(df, pred_model, input_df)
        mlflow.log_params(input_params())
        mlflow.log_metric("prediction", y_hat[0])
    
    st.write("Prediction of Booking cancellation")
    st.write(format_output(y_hat))
