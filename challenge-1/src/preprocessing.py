import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_resize_image(filepath, target_size=(64, 64)):
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize(target_size)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None

def preprocess_data(train_dir, test_dir, train_labels_path, test_ids_path, target_size=(64, 64), test_size=0.2, random_state=42):
    train_df = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(test_ids_path)
    
    test_df.loc[test_df.iloc[:, 0] == 'img_f22972ea.webp', test_df.columns[0]] = 'img_f22972ea.jpg'
    test_df.loc[test_df.iloc[:, 0] == 'img_91cbc6e5.gif', test_df.columns[0]] = 'img_91cbc6e5.png'
    
    train_images = []
    train_labels = []
    for index, row in train_df.iterrows():
        img_path = os.path.join(train_dir, row[train_df.columns[0]])
        img = load_and_resize_image(img_path, target_size)
        if img is not None:
            train_images.append(img.flatten())
            train_labels.append(row[train_df.columns[1]])
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    test_images = []
    test_image_ids = []
    
    if test_df.shape[1] == 1:
        test_df['label'] = 'unknown'
    
    for index, row in test_df.iterrows():
        img_path = os.path.join(test_dir, row[test_df.columns[0]])
        img = load_and_resize_image(img_path, target_size)
        if img is not None:
            test_images.append(img.flatten())
            test_image_ids.append(row[test_df.columns[0]])
    
    test_images = np.array(test_images)
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_labels, 
        test_size=test_size, 
        random_state=random_state
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    
    return X_train, X_val, y_train, y_val, test_images, test_image_ids, label_encoder
