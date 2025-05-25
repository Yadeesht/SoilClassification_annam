import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_data
from src.postprocessing import create_submission_file
import joblib
import argparse

def run_inference(train_dir, test_dir, train_labels_path, test_ids_path, model_path, output_path, model_type='rf'):
    X_train, X_val, y_train, y_val, test_images, test_image_ids, label_encoder = preprocess_data(
        train_dir=train_dir,
        test_dir=test_dir,
        train_labels_path=train_labels_path,
        test_ids_path=test_ids_path
    )

    model = joblib.load(model_path)
    
    if model_type == 'rf':
        test_predictions = model.predict(test_images)
    else:
        test_predictions_encoded = model.predict(test_images)
        test_predictions = label_encoder.inverse_transform(test_predictions_encoded.astype(int))

    create_submission_file(test_image_ids, test_predictions, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Directory containing training images")
    parser.add_argument("--test_dir", required=True, help="Directory containing test images")
    parser.add_argument("--train_labels_path", required=True, help="Path to training labels CSV")
    parser.add_argument("--test_ids_path", required=True, help="Path to test IDs CSV")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--output_path", required=True, help="Path to save predictions")
    parser.add_argument("--model_type", default='rf', choices=['rf', 'xgb'], help="Type of model (rf or xgb)")
    
    args = parser.parse_args()
    
    run_inference(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        train_labels_path=args.train_labels_path,
        test_ids_path=args.test_ids_path,
        model_path=args.model_path,
        output_path=args.output_path,
        model_type=args.model_type
    )
