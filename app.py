from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)

FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def train_model():
    data = pd.read_csv('data_breast_cancer.csv')
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    X = data[FEATURES]
    y = data['diagnosis']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    return train_model()

model = load_model()

@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = [float(request.form.get(feature, 0)) for feature in FEATURES]
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        result = {
            'prediction': 'Malignant' if prediction == 1 else 'Benign',
            'confidence': f"{max(probability) * 100:.2f}%",
            'is_malignant': prediction == 1
        }
        return render_template('index.html', features=FEATURES, result=result)
    except Exception as e:
        return render_template('index.html', features=FEATURES, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
