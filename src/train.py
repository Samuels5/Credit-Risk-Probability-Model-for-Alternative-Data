import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model():
    """
    This function trains, evaluates, and registers the best model.
    """
    # Load the processed data
    try:
        df = pd.read_csv('../data/processed/processed_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run src/data_processing.py first.")
        return

    # Define features (X) and target (y)
    if 'is_high_risk' not in df.columns:
        print("Target variable 'is_high_risk' not found in the processed data.")
        return
        
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define models and hyperparameters for tuning
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear']
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            }
        }
    }

    # Start an MLflow experiment
    mlflow.set_experiment("Credit_Risk_Modeling")

    best_model_run_id = None
    best_model_auc = -1

    for model_name, model_info in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            # Log model name
            mlflow.log_param("model", model_name)

            # Perform Grid Search
            grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Log best parameters
            mlflow.log_params(grid_search.best_params_)

            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log the model
            mlflow.sklearn.log_model(best_model, "model")

            print(f"--- {model_name} ---")
            print(f"ROC AUC: {roc_auc}")
            print(f"Best Parameters: {grid_search.best_params_}")

            # Check if this is the best model so far
            if roc_auc > best_model_auc:
                best_model_auc = roc_auc
                best_model_run_id = run.info.run_id

    # Register the best model
    if best_model_run_id:
        model_uri = f"runs:/{best_model_run_id}/model"
        mlflow.register_model(model_uri, "BestCreditRiskModel")
        print(f"\nRegistered best model from run {best_model_run_id} with ROC AUC {best_model_auc}")


if __name__ == '__main__':
    train_model()