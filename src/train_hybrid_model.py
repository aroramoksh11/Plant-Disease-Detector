import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate
import os

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
# Load Data
# ================================
print("Loading features and labels...")
try:
    deep_features = np.load("data/processed/deep_features.npy")
    handcrafted_features = np.load("data/processed/handcrafted_features.npy")
    y = np.load("data/processed/y.npy")

    print(f"✅ Deep Features Shape: {deep_features.shape}")
    print(f"✅ Handcrafted Features Shape: {handcrafted_features.shape}")
    print(f"✅ Labels Shape: {y.shape}")

    assert deep_features.shape[0] == handcrafted_features.shape[0] == y.shape[0], "Mismatch in number of samples!"

except Exception as e:
    print(f"❌ Error loading features or labels: {e}")
    exit(1)

# ================================
# Build Hybrid Model
# ================================
print("Building hybrid model...")

deep_input = Input(shape=(deep_features.shape[1],), name="deep_input")
handcrafted_input = Input(shape=(handcrafted_features.shape[1],), name="handcrafted_input")

# Concatenate deep and handcrafted features
combined_features = Concatenate()([deep_input, handcrafted_input])

# Classification head
x = Dense(256, activation='relu')(combined_features)
x = Dense(128, activation='relu')(x)
output = Dense(15, activation='softmax')(x)

hybrid_model = Model(inputs=[deep_input, handcrafted_input], outputs=output)
hybrid_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hybrid_model.summary()

# ================================
# Train Hybrid Model
# ================================
print("Training hybrid model...")

history = hybrid_model.fit(
    [deep_features, handcrafted_features], y,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

# ================================
# Save Hybrid Model
# ================================
os.makedirs("models", exist_ok=True)
hybrid_model.save("models/hybrid_model.h5")
print("✅ Hybrid model trained and saved successfully!")

print("🎉 Training complete!")
