import mlflow
import pandas as pd
from fastapi import FastAPI
import uvicorn
import sys
import os

# Add the parent directory to the python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.pydantic_models import PredictionRequest, PredictionResponse

# Set the MLflow tracking URI to connect to the MLflow server
mlflow.set_tracking_uri("http://mlflow:5000")

app = FastAPI(title="Credit Risk Prediction API", description="API for predicting credit risk.")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Load the model from the MLflow Model Registry
        model = mlflow.pyfunc.load_model(model_uri="models:/BestCreditRiskModel/latest")
        print("Model loaded successfully.")
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model is not loaded or available."}
    
    # Convert the request data into a DataFrame
    input_df = pd.DataFrame(request.data)
    
    # Make predictions
    predictions = model.predict(input_df)
    
    # Return the predictions
    return PredictionResponse(predictions=predictions.tolist())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
