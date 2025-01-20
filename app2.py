import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Import cleaned data using pandas
car = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit Web App
st.title("Car Price Prediction Web App")

# Dropdown for car company
companies = sorted(car['company'].unique())
companies.insert(0, 'Select Company')
car_company = st.selectbox("Select Car Company", companies)

# Filter car models based on the selected company
if car_company != 'Select Company':
    car_models = sorted(car[car['company'] == car_company]['name'].unique())
else:
    car_models = []

# Dropdown for car model (only showing models for selected company)
car_model = st.selectbox("Select Car Model", car_models)

# Dropdown for year of manufacture
years = sorted(car['year'].unique(), reverse=True)
year = st.selectbox("Select Year of Manufacture", years)

# Dropdown for fuel type
fuel_types = car['fuel_type'].unique()
fuel_type = st.selectbox("Select Fuel Type", fuel_types)

# Number input for kilometers driven
kms_driven = st.number_input("Enter Kilometers Driven", min_value=0, max_value=1000000, step=500)

# Predict button
if st.button("Predict Price"):
    # Prepare input data for prediction
    input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model, car_company, year, kms_driven, fuel_type]).reshape(1, 5))

    # Prediction using the model
    prediction = model.predict(input_data)

    # Display result
    st.success(f"The predicted price of the car is: ₹ {np.round(prediction[0], 2):,.2f}")
