import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

st.set_page_config(layout="wide")
st.title("Diabetes Dataset Exploration and Prediction")

# Load the saved model from disk
model_1_file_Name = 'model_objects/pipeline1.joblib'
model_2_file_Name = 'model_objects/pipeline2.joblib'
model_3_file_Name = 'model_objects/pipeline3.joblib'
data_df = 'model_objects/diabetes.csv'

# Import saved models
@st.cache_resource
def load_data():
    data_csv = pd.read_csv(data_df)
    return data_csv

@st.cache_resource
def load_model_1():
    return load(model_1_file_Name)

@st.cache_resource
def load_model_2():
    return load(model_2_file_Name)

@st.cache_resource
def load_model_3():
    return load(model_3_file_Name)

df = load_data()
DecisionTree_model = load_model_1()
RandomForest_model = load_model_2()
GradientBoosting_model = load_model_3()

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

ask_to_show_inputs = st.checkbox("Show raw trasformed inputs")

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
    scaled_observations = selected_model[0].transform(new_observations)

    if submit & ask_to_show_inputs:
        st.write('raw inputs:')
        st.write(new_observations)
        st.write('raw transformed inputs:')
        st.write(scaled_observations)
    
    # Make predictions on the new observations
    prediction = selected_model.predict(new_observations)[0]
    prediction_prob = round(np.amax(selected_model.predict_proba(new_observations)) * 100, 2)

    st.write(model_selector, "model predicts:")

    col1, col2 = st.columns(2)
    col1.metric("Outcome", prediction)
    col2.metric("Probability", prediction_prob)

    if prediction == 0:
        st.write("The", model_selector, "model predicts this individual *Does NOT* have diabetes based on the inputs provided on the left pane.")
    else:
        st.write("The", model_selector, "model predicts this individual *Does* have diabetes based on the inputs provided on the left pane.")


    with st.expander("More Details about the modeling steps"):
        st.write("These models were created using scikit-learn library using piplines. The pipline contains standard scalar + the classfier.\
             The models can be improved with hyper-parameters and via additional data exploration techniques (i.e. removeing data points that contains 0 BMI value)")