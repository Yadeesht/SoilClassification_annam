import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_data
from src.postprocessing import evaluate_model, save_metrics, process_predictions
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import argparse

def train_models(train_dir, test_dir, train_labels_path, test_ids_path, output_dir):
    X_train, X_val, y_train, y_val, test_images, test_image_ids, label_encoder = preprocess_data(
        train_dir=train_dir,
        test_dir=test_dir,
        train_labels_path=train_labels_path,
        test_ids_path=test_ids_path
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    val_predictions_rf, test_predictions_rf = process_predictions(
        model=rf_model,
        X_val=X_val,
        test_images=test_images,
        label_encoder=label_encoder,
        model_type='rf'
    )

    metrics_rf = evaluate_model(y_val, val_predictions_rf, model_name="RF")
    save_metrics(metrics_rf, os.path.join(output_dir, "metrics_RF.json"))

    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dval = xgb.DMatrix(X_val, label=y_val_encoded)
    dtest = xgb.DMatrix(test_images)

    params = {
        'objective': 'multi:softmax',
        'num_class': len(label_encoder.classes_),
        'eval_metric': 'merror',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=30, evals=[(dval, 'eval')])

    val_predictions_xgb, test_predictions_xgb = process_predictions(
        model=xgb_model,
        X_val=X_val,
        test_images=test_images,
        label_encoder=label_encoder,
        model_type='xgb'
    )

    metrics_xgb = evaluate_model(y_val, val_predictions_xgb, model_name="XGB")
    save_metrics(metrics_xgb, os.path.join(output_dir, "metrics_XGB.json"))

    return rf_model, xgb_model, label_encoder, test_predictions_rf, test_predictions_xgb, test_image_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Directory containing training images")
    parser.add_argument("--test_dir", required=True, help="Directory containing test images")
    parser.add_argument("--train_labels_path", required=True, help="Path to training labels CSV")
    parser.add_argument("--test_ids_path", required=True, help="Path to test IDs CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_models(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        train_labels_path=args.train_labels_path,
        test_ids_path=args.test_ids_path,
        output_dir=args.output_dir
    )
