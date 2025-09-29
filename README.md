Heart Disease Prediction ML Project
Overview
This project uses machine learning techniques to predict the likelihood of heart disease in patients. The dataset contains patient medical attributes such as age, cholesterol level, blood pressure, and ECG results. The goal is to build and evaluate different models, then select the best-performing one for predictions.

The workflow includes:

Data preprocessing (scaling, PCA).
Training multiple classifiers.
Evaluating models using accuracy and ROC curves.
Visualizing performance.
Deploying a simple Streamlit app for real-time prediction.
Project Structure
heart-disease-ml/
│
├── data/                # Raw and processed datasets
├── models/              # Saved models, scaler, PCA
├── reports/             # Training results and report
├── src/                 # Source code
│   ├── train_models.py
│   ├── evaluate_model.py
│   ├── visualize_results.py
│   ├── generate_report.py
│   └── predict.py
|   └── pca_analysis.py
|   └── model.py
|   └── preprocess_data.py
|   └── save_best_model.py
│
├── app.py               # Streamlit app for prediction
├── requirements.txt     # Dependencies
└── README.md            # Project description
Setup
Clone the repository:

git clone <https://github.com/EsraaSolimanMubarak/-Sprints-x-Microsoft---AI-and-ML-Project.git>
cd heart-disease-ml
Create a virtual environment and install dependencies:

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
Usage
Train models
python src/train_models.py
Evaluate best model
python src/evaluate_model.py
Visualize results
python src/visualize_results.py
Generate report
python src/generate_report.py
Run the Streamlit app
streamlit run app.py
Models
Logistic Regression
Random Forest
XGBoost
The best model is saved in models/best_model.pkl.
