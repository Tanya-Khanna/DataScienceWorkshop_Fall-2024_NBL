# Import necessary libraries and modules

import sys
print(f"Python version: {sys.version}")

# FastAPI is the framework we’re using to build the API
import fastapi
from fastapi import FastAPI, HTTPException

# Pydantic is used here to define data validation and data models for input to the API
import pydantic
from pydantic import BaseModel

# Joblib is a library for saving and loading trained models, used here to load our saved model
import joblib

# Numpy is used for numerical operations, here we might need it to format input data for the model
import numpy as np
import sklearn

print(f"FastAPI version: {fastapi.__version__}")
print(f"Pydantic version: {pydantic.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"joblib version: {joblib.__version__}")

# Load the pre-trained model
# This model was previously trained, saved, and now loaded for use. 
# It's stored in a file named "iris_classifier.pkl", which is a common format for saving Python objects
model = joblib.load("iris_classifier.pkl")

# Initialize the FastAPI app
# This line creates an instance of the FastAPI application.
# `app` will serve as the main entry point for our API, allowing us to define different endpoints.
app = FastAPI()

# Define the input data structure for the API
# We use a Pydantic BaseModel to define a schema for the input data our model requires.
# This schema specifies that our model expects four features, all of which are floating-point numbers (float).
# Each feature represents a specific characteristic of the iris flower, e.g., sepal length, sepal width, etc.
# When an API request is made, FastAPI will validate that the incoming data matches this structure.
class PredictionInput(BaseModel):
    sepal_length: float  # First feature, expecting a float value
    sepal_width: float  # Second feature, expecting a float value
    petal_length: float  # Third feature, expecting a float value
    petal_width: float  # Fourth feature, expecting a float value

# Define the root endpoint
# This is a basic GET endpoint that acts as a welcome message for the API.
# In HTTP, a "GET" request is a type of request used to retrieve data from a server.
# When a user or client sends a GET request to the root URL ("/") of our API, it does not send any data,
# but instead asks the server for a response, typically to retrieve information or display a welcome message.
# Here, the GET request responds with a JSON message saying "Welcome to the Iris Model API!"
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Iris Classification API!",
        "input_format": {
            "sepal_length": "float (cm)",
            "sepal_width": "float (cm)",
            "petal_length": "float (cm)",
            "petal_width": "float (cm)"
        }
    }

# Define the prediction endpoint
# This endpoint allows clients to send data for making predictions.
# Here, we use the "POST" method, meaning clients will send data to the server rather than just retrieve it.
# Clients will send a JSON payload containing the features required by the model (defined in PredictionInput).
# When accessed, this endpoint will take the input data, pass it to the model, and return the prediction.

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Step 1: Convert input data to a numpy array format suitable for the model
        # The input data (from IrisInput) includes four features: sepal_length, sepal_width, petal_length, and petal_width.
        # These values are collected into a list of lists and converted to a numpy array.
        # This structure (2D array) is necessary because the model expects the input in this format.
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Step 2: Make a prediction using the pre-trained model
        # The model uses the 'features' array to predict an iris type.
        # The prediction result is a numeric label that corresponds to one of the iris types.
        prediction = model.predict(features)
        
        # Step 3: Map the numeric prediction to the corresponding iris type
        # The model's output is an integer: 0, 1, or 2. These integers correspond to:
        # 0 -> 'setosa', 1 -> 'versicolor', 2 -> 'virginica'
        # This mapping allows us to provide a meaningful prediction in the API response.
        iris_types = ['setosa', 'versicolor', 'virginica']
        result = iris_types[prediction[0]]
        
        # Step 4: Return the prediction result as a JSON response
        # The response includes two fields:
        # - "prediction": The name of the predicted iris type (e.g., 'setosa')
        # - "prediction_code": The numeric code of the prediction (e.g., 0 for 'setosa')
        return {
            "prediction": result,
            "prediction_code": int(prediction[0])
        }
    
    # Step 5: Handle any errors that may occur during prediction
    # If an error occurs in any of the previous steps, it is caught by this block.
    # An HTTPException with a status code of 500 (internal server error) is raised,
    # and the error message is included in the response to provide diagnostic information.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))