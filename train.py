import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from src.feature_extractor import extract_handcrafted_features
from src.hybrid_model import build_hybrid_model
import os
import gc

# ================================
# 1. Limit TensorFlow Memory Growth
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
# 2. Load Data
# ================================
print("Loading data...")
try:
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")
    print(f"✅ Data loaded! X Shape: {X.shape}, y Shape: {y.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ================================
# 3. Load Pre-trained CNN Model
# ================================
print("Loading CNN model...")
try:
    cnn_model = load_model("models/inception_v3_model.h5")
    
    # Extract features using the global average pooling layer
    feature_extractor = Model(
        inputs=cnn_model.input, 
        outputs=cnn_model.get_layer("global_average_pooling2d").output
    )
    print("✅ CNN Model Loaded!")
except Exception as e:
    print(f"❌ Error loading CNN model: {e}")
    exit(1)

# ================================
# 4. Extract Deep Features
# ================================
print("Extracting deep features...")
try:
    deep_features = feature_extractor.predict(X, batch_size=32, verbose=1)
    print(f"✅ Deep Features Extracted! Shape: {deep_features.shape}")
    np.save("data/processed/deep_features.npy", deep_features)
except Exception as e:
    print(f"❌ Error extracting deep features: {e}")
    exit(1)

# ================================
# 5. Extract Handcrafted Features in Smaller Chunks (Ultimate Fix)
# ================================
print("Extracting handcrafted features in smaller chunks...")

# Reduce chunk size — try 100 or 50
chunk_size = 100  # Lower to avoid memory crashes

try:
    os.makedirs("data/processed/chunks", exist_ok=True)
    
    # Force CPU usage for handcrafted feature extraction
    with tf.device('/CPU:0'):
        for i in range(0, X.shape[0], chunk_size):
            chunk = X[i: i + chunk_size]
            print(f"Processing chunk {i} to {min(i + chunk_size, X.shape[0])}...")

            # Extract handcrafted features for this chunk
            chunk_features = extract_handcrafted_features(chunk)

            # Save chunk features to disk
            np.save(f"data/processed/chunks/handcrafted_features_{i}.npy", chunk_features)

            # Free memory after each chunk
            del chunk, chunk_features
            gc.collect()

    # Combine all chunks into one final file
    print("Combining all handcrafted feature chunks...")
    handcrafted_features_list = [
        np.load(f"data/processed/chunks/handcrafted_features_{i}.npy") 
        for i in range(0, X.shape[0], chunk_size)
    ]
    handcrafted_features = np.vstack(handcrafted_features_list)

    # Save final handcrafted features
    np.save("data/processed/handcrafted_features.npy", handcrafted_features)
    print(f"✅ Handcrafted Features Extracted and Combined! Shape: {handcrafted_features.shape}")

except Exception as e:
    print(f"❌ Error extracting handcrafted features: {e}")
    exit(1)

# ================================
# 6. Validate Feature Shape Consistency
# ================================
if deep_features.shape[0] != handcrafted_features.shape[0]:
    print("❌ Error: Mismatch in number of samples between deep and handcrafted features!")
    exit(1)

# ================================
# 7. Build and Train Hybrid Model
# ================================
print("Building and training hybrid model...")
try:
    hybrid_model = build_hybrid_model(input_shapes=[deep_features.shape[1], handcrafted_features.shape[1]])
    hybrid_model.summary()

    hybrid_model.fit(
        [deep_features, handcrafted_features], y, 
        epochs=10, batch_size=32, verbose=1, validation_split=0.2
    )
    
    # Save the hybrid model
    os.makedirs("models", exist_ok=True)
    hybrid_model.save("models/hybrid_model.h5")
    print("✅ Hybrid model training complete and saved successfully!")

except Exception as e:
    print(f"❌ Error during training: {e}")
    exit(1)
