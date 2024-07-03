# prompt: %%writefile app.py streamlit stunting

%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.joblib')

# Define the application
def main():
    # Create a title and a subheader
    st.title('Stunting Prediction App')
    st.subheader('Please input the following data:')

    # Get the input data from the user
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age_month = st.number_input('Age (Month)', min_value=0, max_value=72)
    body_weight = st.number_input('Body Weight (kg)', min_value=0.0, max_value=20.0)
    body_height = st.number_input('Body Height (cm)', min_value=0.0, max_value=120.0)

    # Preprocess the input data
    data = [[gender, age_month, body_weight, body_height]]
    df = pd.DataFrame(data, columns=['Gender', 'Age (Month)', 'Body weight', 'Body height'])

    # Make predictions
    prediction = model.predict(df)[0]

    # Display the prediction
    st.subheader('Prediction:')
    st.write(prediction)

# Run the application
if __name__ == '__main__':
    main()
