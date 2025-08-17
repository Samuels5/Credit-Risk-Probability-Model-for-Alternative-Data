"""
Model explainability module for the Credit Risk Model.
This module provides functions to explain model predictions using SHAP values.
"""
import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Union, Optional
import joblib
import os

def load_model(model_name: str = "BestCreditRiskModel", stage: str = "latest") -> Any:
    """
    Load a model from the MLflow model registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage to load (latest, production, staging, etc.)
        
    Returns:
        The loaded model
    """
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)

def generate_shap_explainer(model, X: pd.DataFrame) -> shap.Explainer:
    """
    Create a SHAP explainer for the model.
    
    Args:
        model: The trained model
        X: The feature dataset used to initialize the explainer
        
    Returns:
        SHAP explainer object
    """
    # For sklearn models, we need to access the underlying model
    if hasattr(model, 'unwrap_python_model'):
        sklearn_model = model.unwrap_python_model()
    else:
        sklearn_model = model
    
    # Create an appropriate explainer based on model type
    if hasattr(sklearn_model, 'predict_proba'):
        # For models like RandomForest, use TreeExplainer
        if hasattr(sklearn_model, 'estimators_'):
            explainer = shap.TreeExplainer(sklearn_model)
        # For models like LogisticRegression, use LinearExplainer
        else:
            explainer = shap.LinearExplainer(sklearn_model, X)
    else:
        # Default to KernelExplainer for black-box models
        predict_fn = lambda x: sklearn_model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, shap.sample(X, 100))
        
    return explainer

def compute_shap_values(explainer: shap.Explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for the dataset.
    
    Args:
        explainer: SHAP explainer object
        X: Dataset to explain
        
    Returns:
        SHAP values for each feature and sample
    """
    return explainer.shap_values(X)

def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, output_path: Optional[str] = None):
    """
    Create a SHAP summary plot.
    
    Args:
        shap_values: SHAP values from explainer
        X: Feature dataset
        output_path: Path to save the plot image (if None, display the plot)
    """
    plt.figure(figsize=(10, 8))
    
    # For classification models, we often get a list of arrays (one per class)
    if isinstance(shap_values, list):
        # Use values for the positive class (index 1)
        if len(shap_values) > 1:
            shap_values = shap_values[1]
    
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_shap_feature_importance(shap_values: np.ndarray, X: pd.DataFrame, output_path: Optional[str] = None):
    """
    Create a SHAP feature importance plot.
    
    Args:
        shap_values: SHAP values from explainer
        X: Feature dataset
        output_path: Path to save the plot image (if None, display the plot)
    """
    plt.figure(figsize=(10, 6))
    
    # For classification models, we often get a list of arrays (one per class)
    if isinstance(shap_values, list):
        # Use values for the positive class (index 1)
        if len(shap_values) > 1:
            shap_values = shap_values[1]
    
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def explain_prediction(explainer: shap.Explainer, X: pd.DataFrame, instance_idx: int, 
                      output_path: Optional[str] = None):
    """
    Explain a single prediction using SHAP force plot.
    
    Args:
        explainer: SHAP explainer object
        X: Feature dataset
        instance_idx: Index of the instance to explain
        output_path: Path to save the plot image (if None, display the plot)
    """
    # Get the instance to explain
    instance = X.iloc[[instance_idx]]
    
    # Compute SHAP values
    shap_values = compute_shap_values(explainer, instance)
    
    # For classification models, we often get a list of arrays (one per class)
    if isinstance(shap_values, list):
        # Use values for the positive class (index 1)
        if len(shap_values) > 1:
            shap_values = shap_values[1]
    
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list) 
        else explainer.expected_value[1],
        shap_values, 
        instance,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_explainer(explainer: shap.Explainer, filepath: str):
    """
    Save the SHAP explainer for later use.
    
    Args:
        explainer: SHAP explainer object
        filepath: Path to save the explainer
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(explainer, filepath)

def load_explainer(filepath: str) -> shap.Explainer:
    """
    Load a saved SHAP explainer.
    
    Args:
        filepath: Path to the saved explainer
        
    Returns:
        SHAP explainer object
    """
    return joblib.load(filepath)

def generate_explanation_report(model, X: pd.DataFrame, output_dir: str):
    """
    Generate a comprehensive explanation report for the model.
    
    Args:
        model: The trained model
        X: Feature dataset
        output_dir: Directory to save the report assets
    """
    # Create explainer
    explainer = generate_shap_explainer(model, X)
    
    # Compute SHAP values
    shap_values = compute_shap_values(explainer, X)
    
    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Global explanation
    plot_shap_summary(shap_values, X, os.path.join(output_dir, 'shap_summary.png'))
    plot_shap_feature_importance(shap_values, X, os.path.join(output_dir, 'shap_importance.png'))
    
    # Example individual explanations (3 random samples)
    for i in np.random.choice(len(X), 3, replace=False):
        explain_prediction(explainer, X, i, os.path.join(output_dir, f'shap_example_{i}.png'))
    
    # Save the explainer for future use
    save_explainer(explainer, os.path.join(output_dir, 'shap_explainer.joblib'))
    
    print(f"Explanation report generated and saved to {output_dir}")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'output_dir': output_dir
    }
