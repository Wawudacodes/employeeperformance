import numpy as np
import joblib
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
# Loading the trained model using joblib
try:
    loaded_model = joblib.load('./rf_trained_rfmodel.joblib')
except Exception as e:
    loaded_model = None
    logging.error(f"Error loading the model: {e}")

# Prediction function
def performancerating_prediction(input_data, model):
     if model is None:
        logging.error("Model is None. Cannot make prediction.")
        return "Error: Model not loaded"
     
     try:
         # Converting input data to a numpy array
         input_data_as_numpy_array = np.array(input_data, dtype=float)

         # Reshaping the data for prediction
         input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

         # Making a prediction
         prediction = model.predict(input_data_reshaped)

         # Mapping the numeric prediction to performanceratings
         performance_mapping = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
         return performance_mapping.get(prediction[0], "Unknown Rating")
     
     except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return f"Error during prediction: {str(e)}"

# Main function for Streamlit app
def main():

    if loaded_model is None:
        st.error("Model failed to load. Please check the logs for more information.")
        return

    st.title('INX Employee Performance Prediction Web App')

    # Collecting input data
    Age = st.number_input('Employee Age', min_value=18, max_value=65, step=1)
    Gender = st.selectbox('Gender', options=['Male', 'Female'])
    MaritalStatus = st.selectbox('Marital Status', options=['Single', 'Married', 'Divorced'])
    BusinessTravelFrequency = st.selectbox('Business Travel Frequency', options=['Travel_Rarely', 'Travel_Frequently', 'Non_Travel'])
    EmpEducationLevel = st.slider('Education Level (1-5)', min_value=1, max_value=5, step=1)
    EmpEnvironmentSatisfaction = st.slider('Environment Satisfaction (1-4)', min_value=1, max_value=4, step=1)
    EmpJobInvolvement = st.slider('Job Involvement (1-4)', min_value=1, max_value=4, step=1)
    EmpJobSatisfaction = st.slider('Job Satisfaction (1-4)', min_value=1, max_value=4, step=1)
    OverTime = st.selectbox('OverTime', options=['Yes', 'No'])
    EmpLastSalaryHikePercent = st.number_input('Last Salary Hike Percent', min_value=0, max_value=100, step=1)
    EmpRelationshipSatisfaction = st.slider('Relationship Satisfaction (1-4)', min_value=1, max_value=4, step=1)
    TotalWorkExperienceInYears = st.number_input('Total Work Experience (Years)', min_value=0, step=1)
    TrainingTimesLastYear = st.number_input('Training Times Last Year', min_value=0, step=1)
    EmpWorkLifeBalance = st.slider('Work-Life Balance (1-4)', min_value=1, max_value=4, step=1)
    ExperienceYearsAtThisCompany = st.number_input('Experience at Current Company (Years)', min_value=0, step=1)
    ExperienceYearsInCurrentRole = st.number_input('Experience in Current Role (Years)', min_value=0, step=1)
    YearsSinceLastPromotion = st.number_input('Years Since Last Promotion', min_value=0, step=1)
    YearsWithCurrManager = st.number_input('Years With Current Manager', min_value=0, step=1)
    EmpDepartment = st.selectbox('Employee Department', 
                                 options=['Data Science', 'Development', 'Finance', 
                                          'Human Resources', 'Research & Development', 'Sales'])

    # One-hot encode department
    departments = ['Data Science', 'Development', 'Finance', 'Human Resources', 
                   'Research & Development', 'Sales']
    department_encoding = [1 if EmpDepartment == dept else 0 for dept in departments]

    # Mapping categorical inputs to numerical values
    Gender = 0 if Gender == 'Male' else 1
    MaritalStatus = {'Single': 1, 'Married': 2, 'Divorced': 0}[MaritalStatus]
    BusinessTravelFrequency = {'Travel_Rarely': 2, 'Travel_Frequently': 1, 'Non_Travel': 0}[BusinessTravelFrequency]
    OverTime = 1 if OverTime == 'Yes' else 0

    # Combining all inputs
    input_data = [
        Age, Gender, MaritalStatus, BusinessTravelFrequency, EmpEducationLevel, 
        EmpEnvironmentSatisfaction, EmpJobInvolvement, EmpJobSatisfaction, OverTime,
        EmpLastSalaryHikePercent, EmpRelationshipSatisfaction, TotalWorkExperienceInYears,
        TrainingTimesLastYear, EmpWorkLifeBalance, ExperienceYearsAtThisCompany,
        ExperienceYearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
    ] + department_encoding

    # Prediction
    if st.button('Predict Performance Rating'):
        result = performancerating_prediction(input_data, loaded_model)
        if result.startswith("Error"):
            st.error(result)
        else:
            st.success(result)

# Running the app with the main function
if __name__ == '__main__':
    main()
