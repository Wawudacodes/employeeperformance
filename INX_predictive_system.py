import numpy
import pickle

#loading the saved model
loaded_model = pickle.load(open('"C:/Users/User/Documents/IABAC/rf_trained_model.sav', 'rb'))

# Input data (example: EmpLastSalaryHikePercent, EmpEnvironmentSatisfaction, EmpWorkLifeBalance, etc.)
input_data = (1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 2, 1,1,1,1,1)  # Example values for the features used in the model

# Convert input data to a numpy array
input_data_as_numpy_array = np.array(input_data)

# Reshape the data to match the model's expected input shape



input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Assuming rf_model is already trained, make a prediction using the trained Random Forest model
prediction = loaded_model.predict(input_data_reshaped)

# Map the numeric prediction to performance ratings
performance_mapping = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}

if prediction[0] in performance_mapping:
    print(f"The employee's predicted performance rating is: {performance_mapping[prediction[0]]}")
else:
    print("Prediction could not be mapped to a performance rating.")
