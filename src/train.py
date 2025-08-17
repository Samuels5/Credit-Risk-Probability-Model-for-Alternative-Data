import pandas as pd
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.model_explainer import generate_explanation_report

def train_model():
    """
    This function trains, evaluates, and registers the best model.
    """
    # Load the processed data
    try:
        df = pd.read_csv('Data/processed/processed_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run src/data_processing.py first.")
        return

    # Define features (X) and target (y)
    if 'is_high_risk' not in df.columns:
        print("Target variable 'is_high_risk' not found in the processed data.")
        return
        
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']

    # Drop identifier columns that are not features
    if 'CustomerId' in X.columns:
        X = X.drop('CustomerId', axis=1)
    if 'TransactionStartTime' in X.columns:
        X = X.drop('TransactionStartTime', axis=1)

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
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Credit_Risk_Modeling")

    best_model_run_id = None
    best_model_auc = -1
    best_model_obj = None  # Store the actual model object for later use

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

            # Create and log performance visualizations
            # Confusion Matrix
            os.makedirs('artifacts/model_evaluation', exist_ok=True)
            cm_fig_path = f'artifacts/model_evaluation/{model_name}_confusion_matrix.png'
            plot_confusion_matrix(y_test, y_pred, cm_fig_path)
            mlflow.log_artifact(cm_fig_path, "evaluation_plots")
            
            # ROC Curve
            roc_fig_path = f'artifacts/model_evaluation/{model_name}_roc_curve.png'
            plot_roc_curve(y_test, y_pred_proba, roc_fig_path)
            mlflow.log_artifact(roc_fig_path, "evaluation_plots")
            
            # Precision-Recall Curve
            pr_fig_path = f'artifacts/model_evaluation/{model_name}_pr_curve.png'
            plot_precision_recall_curve(y_test, y_pred_proba, pr_fig_path)
            mlflow.log_artifact(pr_fig_path, "evaluation_plots")

            # Log the model
            mlflow.sklearn.log_model(best_model, "model")

            print(f"--- {model_name} ---")
            print(f"ROC AUC: {roc_auc}")
            print(f"Best Parameters: {grid_search.best_params_}")

            # Check if this is the best model so far
            if roc_auc > best_model_auc:
                best_model_auc = roc_auc
                best_model_run_id = run.info.run_id
                best_model_obj = best_model

    # Register the best model
    if best_model_run_id:
        model_uri = f"runs:/{best_model_run_id}/model"
        mlflow.register_model(model_uri, "BestCreditRiskModel")
        print(f"\nRegistered best model from run {best_model_run_id} with ROC AUC {best_model_auc}")
        
        # Generate SHAP explanations for the best model
        if best_model_obj:
            print("\nGenerating model explanations using SHAP...")
            explanation_dir = 'artifacts/model_explanations'
            explanation_results = generate_explanation_report(best_model_obj, X_test, explanation_dir)
            
            # Log the explanation artifacts to MLflow
            for file in os.listdir(explanation_dir):
                if file.endswith('.png'):
                    mlflow.log_artifact(os.path.join(explanation_dir, file), "model_explanations")
            
            print("Model explanations have been generated and logged to MLflow")
    
    return best_model_obj, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Create and save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Not High Risk', 'High Risk']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_score, output_path):
    """Create and save an ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_score, output_path):
    """Create and save a precision-recall curve visualization."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    train_model()