pip install streamlit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings

# Filter out warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load data
data = pd.read_csv('C:/Users/Sonuu/OneDrive/Desktop/desktop/ml project/creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write(f"Model Training Accuracy: {train_acc:.2f}")
st.write(f"Model Testing Accuracy: {test_acc:.2f}")

st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features (comma-separated values for each feature)')

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    if input_df:
        # Get input feature values
        input_df_lst = input_df.split(',')
        
        # Validate that the number of features matches the model's expectation
        if len(input_df_lst) != X_train.shape[1]:
            st.write(f"Error: Please enter exactly {X_train.shape[1]} features.")
        else:
            try:
                # Convert the input values to float and reshape for prediction
                features = np.array(input_df_lst, dtype=np.float64).reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(features)
                
                # Display result
                if prediction[0] == 0:
                    st.write("Legitimate transaction")
                else:
                    st.write("Fraudulent transaction")
            except ValueError:
                st.write("Error: Please enter numeric values for all features.")
    else:
        st.write("Please enter the feature values.")
