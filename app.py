import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.joblib')

# Define the application
def main():
    # Create the title and sidebar
    st.title('Stunting Prediction App')
    st.sidebar.header('User Input Features')

    # Get user input
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    age_month = st.sidebar.number_input('Age (Month)')
    body_weight = st.sidebar.number_input('Body Weight (kg)')
    body_height = st.sidebar.number_input('Body Height (cm)')

    # Encode categorical data
    gender_encoded = 0 if gender == 'Female' else 1

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Gender': [gender_encoded],
        'Age (Month)': [age_month],
        'Body weight': [body_weight],
        'Body height': [body_height]
    })

    # Normalize numerical data
    minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    user_input['Body height'] = minmax.fit_transform(user_input['Body height'].values.reshape(-1, 1))

    # Predict the status
    prediction = model.predict(user_input)[0]

    # Display the prediction
    if st.button('Predict'):
        st.subheader('Prediction:')
        st.write(prediction)

# Run the application
if __name__ == '__main__':
    main()
