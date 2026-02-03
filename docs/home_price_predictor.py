import pandas as pd  
import streamlit as st 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
from xgboost import XGBRegressor  
from sklearn.model_selection import train_test_split  
import logging
import matplotlib.pyplot as plt
from datetime import datetime 
import os
import openai 
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

#Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#App Title
st.title("Home Price Predictor")
    
#Sidebar controls
st.header("Upload Your CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#Load default dataset, then replace with uploaded one if present
df = pd.read_csv('Housing.csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
#Logging file
LOG_FILE = "log.csv"

def append_log(record: dict):
    log_df = pd.DataFrame([record])
    if os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, index=False)
        
def show_error(msg: str, exc: Exception | None = None):
    """Display and log errors in a single call."""
    if exc:
        logging.exception(msg)
    else:
        logging.error(msg)
    st.error(f"{msg}")

#Feature selection
st.header("Select Features")
all_columns = df.columns.tolist()
input_columns = st.multiselect("Select input features", all_columns)

#Output column
output_column = "price"
st.text(f"Output Feature: {output_column}")

#Test size slider
test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100

#Model selection
model_choice = st.selectbox('Choose a Regression Model', ["Linear Regression", "XGBoost"])
if model_choice == 'Linear Regression':
    model = LinearRegression()
else:
    model = XGBRegressor(objective="reg:squarederror", n_jobs=1)

#Train model button
train_now = st.button("Train Model")

#Main Section
st.subheader("Dataset Preview")
st.write(df.head())

if input_columns and train_now:
    try:
        #Check output columns
        if output_column not in df.columns:
            st.error(f"Output column '{output_column}' not found.")
            st.stop()

        #Prepare input and output data
        X = pd.get_dummies(df[input_columns], drop_first=True).fillna(0)
        y = pd.to_numeric(df[output_column], errors="coerce").fillna(0)

        if model_choice == "XGBoost" and not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
            st.error("All input features must be numeric for XGBoost.")
            st.stop()

        #Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        append_log({
        "timestamp": datetime.now(),
        "event": "train",
        "model": model_choice,
        "features": "|".join(input_columns),
        "mse": mse
    })
        
        #Graphs
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        #Displays actual vs predicted graphs
        st.subheader("Actual vs. Predicted Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # identity line
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs. Predicted House Prices")
        st.pyplot(fig)
        

        #Save themodel to session state
        st.session_state.trained_model = model
        st.session_state.feature_columns = X.columns

        #Display results
        st.success("Model trained successfully!")
        st.write(f"**Mean Squared Error (test set):** {mse:.2f}")

        if model_choice == "Linear Regression":
            st.write("**Coefficients:**")
            st.write(pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_}))
            st.write(f"**Intercept:** {model.intercept_:.2f}")

    except Exception as e:
        logging.exception("Training failed")
        st.error(f"Training error: {e}")
elif train_now and not input_columns:
    st.warning("Please select at least one input feature before training.")

#Prediction Section 
if "trained_model" in st.session_state:
    st.subheader("Make a Prediction")

    user_vals = []
    for col in input_columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                default_val = float(df[col].mean())
            else:
                default_val = 0.0
            val = st.number_input(f"{col}", value=default_val)
        except:
            val = st.number_input(f"{col}", value=0.0)
        user_vals.append(val)

    if st.button("Predict"):
        try:
            user_df = pd.DataFrame([user_vals], columns=input_columns)
            user_df = pd.get_dummies(user_df)
            user_df = user_df.reindex(columns=st.session_state.feature_columns, fill_value=0)

            pred = st.session_state.trained_model.predict(user_df)[0]
            st.success(f"Predicted {output_column}: {pred:.2f}")
        except Exception as e:
            logging.exception("Prediction failed")
            st.error(f"Prediction error: {e}")
else:
    st.info("Train the model to enable predictions.")
    
st.sidebar.header("Upload Any File")
uploaded_extra_file = st.sidebar.file_uploader("Upload a file", type=None)  # type=None allows all types

df_extra = None
if uploaded_extra_file is not None and uploaded_extra_file.name.endswith(".csv"):
    try:
        df_extra = pd.read_csv(uploaded_extra_file)
        st.sidebar.write(f"Loaded extra file: {uploaded_extra_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error reading extra CSV: {e}")

if uploaded_extra_file is not None:
    st.sidebar.write(f"Uploaded file: {uploaded_extra_file.name}")
    

st.sidebar.subheader("Chat with an AI Assistant")
st.sidebar.info('Ask a question about this application or your dataset')
user_question = st.sidebar.text_input(" Ask a question:")

if user_question: 
    with st.spinner("Thinking..."):
        try: 
            df_sample = df.head(10).to_string(index=False),
            df_columns = ', '. join(df.columns)
            df_shape = df.shape
        
            system_msg = (f"You are an assistant inside a home price prediction application. "
                f"The dataset has {df_shape[0]} rows and {df_shape[1]} columns. "
                f"The columns are: {df_columns}. "
                f"Here is a summary of the dataset:\n{df_sample}\n"
                f"Answer questions about the dataset, machine learning model, and predictions clearly."
            )
            if df_extra is not None:
                df_sample_extra = df_extra.head(5).to_string(index=False)
                df_columns_extra = ", ".join(df_extra.columns)
                df_shape_extra = df_extra.shape

                system_msg += (
                    f"Second uploaded dataset:\n"
                    f"- Shape: {df_shape_extra[0]} rows × {df_shape_extra[1]} columns\n"
                    f"- Columns: {df_columns_extra}\n"
                    f"- Sample rows:\n{df_sample_extra}\n\n"
                )

            system_msg += "Answer questions about either dataset, including specific rows or comparisons."

            response = openai.chat.completions.create(
                model = 'gpt-4o',
                messages = [
                    {'role': 'system', 'content': system_msg},
                     {'role': 'user', 'content': user_question}
                ]
            )
            answer = response.choices[0].message.content
            st.sidebar.success(answer)
        except Exception as e:
            st.error(f"OpenAI Error: {e}")
            
    #Download Logs
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                label="Download Logs",
                data=f,
                file_name="model_log.csv",
                mime="text/csv"
            )
