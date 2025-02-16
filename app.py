# app.py
import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load the trained models
def load_model(disease):
    with open(f'models/{disease}_model.pkl', 'rb') as file:
        return pickle.load(file)

# Function to make predictions
def predict_disease(model, features):
    prediction = model.predict([features])
    return prediction[0]

def main():
    st.title("Disease Prediction System")
    
    # Sidebar for navigation
    disease_type = st.sidebar.selectbox(
        "Select Disease to Predict",
        ["Diabetes", "Heart Disease", "Parkinson's Disease"]
    )
    
    if disease_type == "Diabetes":
        st.header("Diabetes Prediction")
        # Diabetes input fields
        glucose = st.number_input("Glucose Level", 0, 200, 100)
        bp = st.number_input("Blood Pressure", 0, 122, 72)
        insulin = st.number_input("Insulin", 0, 846, 80)
        bmi = st.number_input("BMI", 0.0, 67.1, 25.0)
        age = st.number_input("Age", 0, 120, 30)
        
        features = [glucose, bp, insulin, bmi, age]
        
        if st.button("Predict Diabetes"):
            model = load_model('diabetes')
            result = predict_disease(model, features)
            if result == 1:
                st.error("High risk of diabetes detected!")
            else:
                st.success("Low risk of diabetes detected.")
                
    elif disease_type == "Heart Disease":
        st.header("Heart Disease Prediction")
        # Heart disease input fields
        age = st.number_input("Age", 0, 120, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
        trestbps = st.number_input("Resting Blood Pressure", 0, 200, 120)
        chol = st.number_input("Cholesterol", 0, 600, 200)
        
        features = [age, 1 if sex == "Male" else 0, cp, trestbps, chol]
        
        if st.button("Predict Heart Disease"):
            model = load_model('heart')
            result = predict_disease(model, features)
            if result == 1:
                st.error("High risk of heart disease detected!")
            else:
                st.success("Low risk of heart disease detected.")
                
    else:
        st.header("Parkinson's Disease Prediction")
        # Parkinson's disease input fields
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", 0.0, 300.0, 120.0)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 600.0, 200.0)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", 0.0, 300.0, 100.0)
        mdvp_jitter = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.0)
        
        features = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter]
        
        if st.button("Predict Parkinson's"):
            model = load_model('parkinsons')
            result = predict_disease(model, features)
            if result == 1:
                st.error("High risk of Parkinson's disease detected!")
            else:
                st.success("Low risk of Parkinson's disease detected.")

if __name__ == "__main__":
    main()

# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Train Diabetes Model
def train_diabetes_model():
    data = pd.read_csv('data/diabetes.csv')
    X = data[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    with open('models/diabetes_model.pkl', 'wb') as file:
        pickle.dump(model, file)

