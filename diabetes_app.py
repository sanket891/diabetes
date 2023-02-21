import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")
st.title("Diabetes Dataset Exploration with Prediction")

# Load the saved model from disk
model_1_file_Name = 'model_objects/pima_indians_diabetes_DecisionTree_model.pkl'
model_2_file_Name = 'model_objects/pima_indians_diabetes_RandomForest_model.pkl'
model_3_file_Name = 'model_objects/pima_indians_diabetes_GradientBoosting_model.pkl'
scaler_name = 'model_objects/scaler_saved.pkl'
data_df = 'model_objects/diabetes.csv'

# Import saved models
@st.experimental_singleton
def load_data():
    data_csv = pd.read_csv(data_df)
    return data_csv

@st.experimental_singleton
def load_model_1():
    model_1 = pickle.load(open(model_1_file_Name, 'rb'))
    return model_1

@st.experimental_singleton
def load_model_2():
    model_2 = pickle.load(open(model_2_file_Name, 'rb'))
    return model_2

@st.experimental_singleton
def load_model_3():
    model_3 = pickle.load(open(model_3_file_Name, 'rb'))
    return model_3

@st.experimental_singleton
def load_model_4():
    model_4 = pickle.load(open(scaler_name, 'rb'))
    return model_4

df = load_data()
DecisionTree_model = load_model_1()
RandomForest_model = load_model_2()
GradientBoosting_model = load_model_3()
scaler = load_model_4()

# Define the main function of the Streamlit app

st.subheader("Pima Indians Diabetes Data exploration")

if st.checkbox("Show Dataset"):
    st.dataframe(df)

if st.checkbox("Show Statistics"):
    st.write(df.describe())

if st.checkbox("Correlation Matrix"):
    st.write(df.corr())

if st.checkbox("Pre-Trained Models Stats"):
    st.write("DecisionTreeClassifier -- Accuracy: 75.32%")
    st.write("RandomForestClassifier -- Accuracy: 77.92%")
    st.write("GradientBoostingClassifier -- Accuracy: 78.79%")

#model selector
st.subheader("Predict")
model_selector = st.sidebar.radio("Please Pick a model for prediction", ('DecisionTree', 'RandomForest', 'GradientBoosting'))

#switch models based on model selection radio option
if model_selector == "DecisionTree":
    selected_model = DecisionTree_model
    
elif model_selector == "RandomForest":
    selected_model = RandomForest_model

elif model_selector == "GradientBoosting":
    selected_model = GradientBoosting_model

else:
    selected_model = None

# Create a form to input the features
pregnancies = st.sidebar.number_input("Number of pregnancies:", min_value=0, max_value=17,value = 6)
glucose = st.sidebar.number_input("Plasma glucose concentration:", min_value=0, max_value=199, value = 148)
blood_pressure = st.sidebar.number_input("Diastolic blood pressure (mm Hg):", min_value=0, max_value=122, value = 72)
skin_thickness = st.sidebar.number_input("Triceps skin fold thickness (mm):", min_value=0, max_value=99, value = 35)
insulin = st.sidebar.number_input("2-Hour serum insulin (mu U/ml):", min_value=0, max_value=846, value = 0)
bmi = st.sidebar.number_input("BMI (weight in kg/(height in m)^2):", min_value=0.0, max_value=67.1, value = 33.6)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes pedigree function:", min_value=0.000, max_value=2.420, value = 0.627, step= 0.001, format='%f')
age = st.sidebar.number_input("Age (years):", min_value=0, max_value=81, value = 50)

# Create a submit button
submit = st.sidebar.button("Predict")

if submit:
    # Standardize the new observations
    new_observations = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    scaled_observations = scaler.transform(new_observations)

    st.write(new_observations)
    st.write(scaled_observations)
    
    # Make predictions on the new observations
    prediction = selected_model.predict(scaled_observations)[0]
    prediction_prob = round(np.amax(selected_model.predict_proba(new_observations)) * 100, 2)

    if prediction == 0:
        st.write("No, the", model_selector, "model predicts this individual *Does NOT* have diabetes based on the inputs provides on the laft pane")
    else:
        st.write("Yes, the", model_selector, "model predicts this individual *Does* have diabetes based on the inputs provides on the laft pane")

    st.write(model_selector, "model predicts:", prediction, "as outcome with", prediction_prob, "% Probability")