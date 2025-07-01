import pandas as pd
import mlflow
import argparse
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import preprocess_data

def make_predictions(input_filepath: str, output_filepath: str):
    """
    Loads the model, preprocesses new data, makes predictions, and saves them.
    """
    print("Loading model...")
    try:
        # Load the model from the MLflow Model Registry
        model = mlflow.pyfunc.load_model(model_uri="models:/BestCreditRiskModel/latest")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading new data from {input_filepath}...")
    try:
        new_data = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return

    # Keep customer IDs for the final output
    customer_ids = new_data['CustomerId']

    print("Preprocessing new data...")
    # The preprocess_data function needs to be adapted or called carefully
    # as it was designed to also create the target variable, which is not present here.
    # For now, we assume it can handle data without the target for prediction.
    try:
        # We need to ensure the target variable engineering part doesn't break this
        # A robust implementation would separate fitting of preprocessors from transformation
        processed_new_data = preprocess_data(new_data.copy())
        # Drop the target if it was created during preprocessing
        if 'is_high_risk' in processed_new_data.columns:
            processed_new_data = processed_new_data.drop('is_high_risk', axis=1)
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return

    print("Making predictions...")
    # Drop the columns that were not used during training
    processed_new_data_for_prediction = processed_new_data.drop(columns=['CustomerId', 'TransactionStartTime'], errors='ignore')
    predictions = model.predict(processed_new_data_for_prediction)

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'CustomerId': customer_ids,
        'Risk_Probability': predictions
    })

    print(f"Saving predictions to {output_filepath}...")
    results_df.to_csv(output_filepath, index=False)
    print("Predictions saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions on new data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file with new data.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV file with predictions.")
    args = parser.parse_args()
    make_predictions(args.input, args.output)
