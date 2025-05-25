import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_data
from src.postprocessing import evaluate_model, save_metrics, process_predictions
from sklearn.ensemble import IsolationForest
import argparse

def train_model(train_dir, test_dir, train_labels_path, test_ids_path, output_dir):
    X_train, X_val, y_train, y_val, test_images, test_image_ids = preprocess_data(
        train_dir=train_dir,
        test_dir=test_dir,
        train_labels_path=train_labels_path,
        test_ids_path=test_ids_path
    )

    model = IsolationForest(contamination=0.18)
    model.fit(X_train)

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(test_images)

    val_predictions = process_predictions(val_predictions)
    test_predictions = process_predictions(test_predictions)

    metrics = evaluate_model(y_val, val_predictions, model_name="IF")
    save_metrics(metrics, os.path.join(output_dir, "metrics_IF.json"))

    return model, test_predictions, test_image_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Directory containing training images")
    parser.add_argument("--test_dir", required=True, help="Directory containing test images")
    parser.add_argument("--train_labels_path", required=True, help="Path to training labels CSV")
    parser.add_argument("--test_ids_path", required=True, help="Path to test IDs CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_model(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        train_labels_path=args.train_labels_path,
        test_ids_path=args.test_ids_path,
        output_dir=args.output_dir
    )
