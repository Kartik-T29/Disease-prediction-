from ast import main


if __name__ == "__main__":
    main()

# train_models. 
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