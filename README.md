Home Price Predictor
This project is a Streamlit web app that lets users explore housing data, train machine learning models, and predict home prices in an interactive way. The goal was to build something hands-on that shows the full workflow of a simple ML project, from data upload to model evaluation and prediction.
Rather than hard-coding a single dataset or feature set, the app is designed to be flexible so users can experiment with different inputs and see how model performance changes.
What the App Does
Loads a default housing dataset or lets users upload their own CSV
Displays a preview of the dataset so users can see available columns
Allows users to select which features to use for prediction
Trains either a Linear Regression model or an XGBoost regression model
Lets users control the train/test split size
Evaluates model performance using Mean Squared Error (MSE)
Shows a plot comparing actual vs. predicted home prices
Displays coefficients when using Linear Regression
Allows users to enter custom values and generate a price prediction
Logs training runs and allows logs to be downloaded
Includes a sidebar AI assistant to answer questions about the data or model
Why I Built This
I built this project to better understand how machine learning models behave when feature selection, model choice, and data inputs change. I also wanted to practice turning a machine learning workflow into an actual interactive tool instead of keeping everything inside a notebook.
The focus was on usability, experimentation, and learning rather than building a production-ready system.
Tech Stack
Python
Streamlit
Pandas
Scikit-learn
XGBoost
Matplotlib
OpenAI API (for the in-app assistant)
Dataset Requirements
The dataset must be a CSV file
It must include a column named price, which is used as the prediction target
Other columns can be numeric or categorical
Categorical features are automatically one-hot encoded during training
How to Run the App
Clone the repository
(Optional) Create and activate a virtual environment
Install dependencies:
pip install -r requirements.txt
Create a .env file and add your OpenAI API key:
API_KEY=your_api_key_here
Run the app:
streamlit run app.py
Notes and Limitations
The output variable is currently fixed as price
Prediction inputs are numeric only
Training logs are stored locally in a CSV file
This project is intended for learning and experimentation
Possible Improvements
Let users choose the output column
Better handling of categorical inputs during prediction
Add cross-validation
Save and reload trained models
Deploy the app publicly using Streamlit Cloud
Author
Rhea Mammen
University of Maryland
Information Science & Sociology
