# Disease Prediction System

A web-based application that predicts the likelihood of various diseases including Diabetes, Heart Disease, and Parkinson's Disease using machine learning models.

## Features

- Prediction for multiple diseases:
  - Diabetes
  - Heart Disease
  - Parkinson's Disease
- User-friendly web interface built with Streamlit
- Real-time predictions
- High accuracy using Random Forest Classifier

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kartik-T29/Disease-prediction-.git
cd disease-prediction-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Information

The system uses three different datasets:

1. Diabetes Dataset:
   - Features: Glucose, Blood Pressure, Insulin, BMI, Age
   - Target: Diabetes presence (0 or 1)
   - Source: PIMA Indians Diabetes Database

2. Heart Disease Dataset:
   - Features: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol
   - Target: Heart disease presence (0 or 1)
   - Source: UCI Machine Learning Repository

3. Parkinson's Disease Dataset:
   - Features: Various voice measure parameters including MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%)
   - Target: Parkinson's disease presence (0 or 1)
   - Source: UCI Machine Learning Repository

## Usage

1. Train the models first:
```bash
python train_models.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and go to `http://localhost:8501`

4. Select the disease you want to predict from the sidebar and input the required parameters

## Model Information

The system uses Random Forest Classifier for all three disease predictions. The models are trained on standardized data with an 80-20 train-test split.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

