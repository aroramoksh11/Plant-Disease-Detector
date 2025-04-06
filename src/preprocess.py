import os
import cv2
import numpy as np

# Constants
IMG_SIZE = 224
DATASET_PATH = "data/train"
PROCESSED_PATH = "data/processed"

def preprocess_images(dataset_path):
    images, labels = [], []

    # Get all class names and sort alphabetically
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    # Map class names to indices (0 to 14)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    print(f"✅ Found {len(class_names)} classes: {class_to_idx}")

    # Preprocess images
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"⚠️ Skipping non-image file: {img_name}")
                continue

            # Read and resize image
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Warning: Could not read {img_path}. Skipping...")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize

            # Append image and labels
            images.append(img)
            labels.append(class_to_idx[class_name])

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Confirm processing
    print(f"✅ Preprocessing complete! Total images: {len(X)}")

    # Save preprocessed data
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    np.save(f"{PROCESSED_PATH}/X.npy", X)
    np.save(f"{PROCESSED_PATH}/y.npy", y)
    print("✅ Data saved successfully!")

    return X, y

# Run preprocessing
X, y = preprocess_images(DATASET_PATH)
