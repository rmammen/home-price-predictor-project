# Home Price Predictor  
Interactive Machine Learning Web App

---

## Overview
Home Price Predictor is a Streamlit web application that allows users to explore housing datasets, train machine learning models, and predict home prices. The app is designed to be flexible and interactive, making it easy to experiment with different features, models, and data inputs without writing additional code.

This project focuses on learning and experimentation rather than production deployment.

---

## Project Goals
- Practice building an end-to-end machine learning workflow  
- Turn a notebook-based ML process into an interactive application  
- Compare baseline and advanced regression models  
- Improve understanding of feature selection and model evaluation  
- Build a user-friendly interface for data exploration  

---

## Key Features
- Upload a housing CSV or use the default dataset  
- Preview the dataset to inspect columns and values  
- Dynamically select input features  
- Train either:
  - Linear Regression (interpretable baseline)
  - XGBoost Regressor (non-linear model)
- Adjust the train/test split size  
- Evaluate performance using Mean Squared Error (MSE)  
- Visualize actual vs. predicted prices  
- Display model coefficients for Linear Regression  
- Make custom predictions after training  
- Log training runs and download logs as CSV  
- Sidebar AI assistant to answer questions about the data and models  

---

## Dataset Requirements
- Data must be in CSV format  
- A column named `price` is required as the target variable  
- Other columns may be numeric or categorical  
- Categorical features are automatically one-hot encoded during training  

---

## How to Use the App

### 1. Run the App
Run the following command from the project directory:

    streamlit run app.py

### 2. Load a Dataset
- A default housing dataset loads automatically  
- Optionally upload your own CSV file using the upload control  

### 3. Preview the Data
- The first few rows of the dataset are displayed to confirm the file loaded correctly  

### 4. Select Features
- Choose which columns to use as input features  
- The output variable is fixed as `price`  

### 5. Choose a Model
- Select either Linear Regression or XGBoost from the dropdown  

### 6. Adjust Train/Test Split
- Use the slider to control how much data is reserved for testing  

### 7. Train the Model
- Click the Train Model button  
- View the Mean Squared Error and the actual vs. predicted plot  

### 8. Review Results
- If using Linear Regression, model coefficients and intercept are displayed  

### 9. Make Predictions
- Enter custom feature values  
- Click Predict to generate a price estimate  

### 10. Use the AI Assistant (Optional)
- Ask questions about the dataset, features, or model behavior in the sidebar  

### 11. Download Logs
- Training logs can be downloaded as a CSV file for review  

---

## Logging and Experiment Tracking
Each training run is logged with:
- Timestamp  
- Selected model  
- Chosen input features  
- Mean Squared Error  

Logs are stored locally and can be downloaded from the app.

---

## Limitations
- Output column is currently fixed as `price`  
- Prediction inputs are numeric only  
- Logs are stored locally per session  
- Not intended for production use  

---

## Future Improvements
- Allow dynamic selection of the output column  
- Improve handling of categorical inputs during prediction  
- Add cross-validation  
- Save and reload trained models  
- Deploy the app using Streamlit Cloud  

---

## Author
Rhea Mammen  
University of Maryland  
Information Science and Sociology
