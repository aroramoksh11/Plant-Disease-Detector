import numpy as np
import tensorflow as tf
import os
import gc
from src.feature_extractor import extract_handcrafted_features

# ================================
# Limit TensorFlow Memory Growth
# ================================
print("Setting TensorFlow memory growth...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow memory growth set!")
    except RuntimeError as e:
        print(f"❌ Error setting memory growth: {e}")

# ================================
# Load Data (Full Dataset)
# ================================
print("Loading data...")
try:
    X = np.load("data/processed/X.npy")
    print(f"✅ Data loaded! X Shape: {X.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ================================
# Extract Handcrafted Features (One Image at a Time)
# ================================
print("Extracting handcrafted features...")

try:
    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    # Force CPU to avoid GPU memory overload
    with tf.device('/CPU:0'):
        handcrafted_features_list = []

        # Process each image individually
        for i in range(X.shape[0]):
            print(f"Processing image {i + 1}/{X.shape[0]}...")

            # Extract features for one image
            features = extract_handcrafted_features(X[i])

            # Ensure features are flattened into 1D array
            features = np.array(features).flatten()

            handcrafted_features_list.append(features)

            # Manual memory cleanup
            del features
            gc.collect()

    # Combine all features and save
    handcrafted_features = np.vstack(handcrafted_features_list)
    np.save("data/processed/handcrafted_features.npy", handcrafted_features)
    print(f"✅ Handcrafted Features Extracted and Saved! Shape: {handcrafted_features.shape}")

except Exception as e:
    print(f"❌ Error extracting handcrafted features: {e}")
    exit(1)
