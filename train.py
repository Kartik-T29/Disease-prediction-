import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def train_diabetes_model():
    print("Training Diabetes Model...")
    data = pd.read_csv('data/diabetes.csv')
    X = data[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and print accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy:.2%}")
    
    with open('models/diabetes_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Diabetes Model saved successfully\n")

def train_heart_model():
    print("Training Heart Disease Model...")
    data = pd.read_csv('data/heart.csv')
    X = data[['age', 'sex', 'cp', 'trestbps', 'chol']]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and print accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {accuracy:.2%}")
    
    with open('models/heart_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Heart Disease Model saved successfully\n")

def train_parkinsons_model():
    print("Training Parkinson's Disease Model...")
    data = pd.read_csv('data/parkinsons.csv')
    X = data[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)']]
    y = data['status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and print accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Parkinson's Disease Model Accuracy: {accuracy:.2%}")
    
    with open('models/parkinsons_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Parkinson's Disease Model saved successfully\n")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")
    
    print("Starting model training process...\n")
    train_diabetes_model()
    train_heart_model()
    train_parkinsons_model()
    print("All models trained and saved successfully!")