# Breast Cancer Prediction Web App

A machine learning web application that predicts whether a breast tumor is **malignant** or **benign** based on 30 clinical features. Built with Python, Scikit-learn, and Flask.

## Live Demo

Enter 30 clinical measurements and get an instant prediction with confidence score.

## Features

- Compares 5 ML models: Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN
- Best model (Random Forest) achieves **98% accuracy**
- Flask-based web interface to process 30 clinical features
- Real-time prediction with confidence score
- Clean, responsive UI across all devices

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn, Pandas, NumPy |
| Web | Flask, HTML, CSS |
| Visualization | Matplotlib, Seaborn |
| Dataset | Wisconsin Breast Cancer Dataset |

## Project Structure

```
Breast_cancer_prediction/
├── app.py                  # Flask web application
├── model.py                # ML model training & comparison script
├── data_breast_cancer.csv  # Dataset
├── requirements.txt        # Dependencies
└── templates/
    └── index.html          # Web UI
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Shrutiraj26/Breast_cancer_prediction.git
cd Breast_cancer_prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the web app**
```bash
python app.py
```

**4. Open in browser**
```
http://localhost:5000
```

## Model Comparison

| Model | Accuracy |
|---|---|
| Random Forest | 98% |
| Gradient Boosting | 97% |
| SVM | 97% |
| Logistic Regression | 96% |
| KNN | 95% |

Run `model.py` to reproduce the full comparison with graphs, ROC curves, and feature importance plots.

## Dataset

The Wisconsin Breast Cancer Dataset contains 569 samples with 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.

- **Benign:** 357 samples
- **Malignant:** 212 samples
