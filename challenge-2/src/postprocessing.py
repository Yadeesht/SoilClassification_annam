import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import os

def evaluate_model(y_true, y_pred, model_name="model"):
    metrics = {
        f"{model_name}_validation_weighted_f1_score": f1_score(y_true, y_pred, average='weighted'),
        f"{model_name}_validation_weighted_precision": precision_score(y_true, y_pred, average='weighted'),
        f"{model_name}_validation_weighted_recall": recall_score(y_true, y_pred, average='weighted')
    }
    return metrics

def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def create_submission_file(test_image_ids, predictions, output_path):
    submission_df = pd.DataFrame({
        'image_id': test_image_ids,
        'soil_type': predictions
    })
    submission_df.to_csv(output_path, index=False)

def process_predictions(predictions):
    return np.where(predictions == -1, 0, predictions)
