import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# ================================
# Force CPU Execution (Disable Metal)
# ================================

# ================================
# Hardcoded Disease Names
# ================================
DISEASE_LABELS = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca", "Grape - Leaf Blight", "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot", "Peach - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites", "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", "Tomato - Healthy"
]

# ================================
# Load Data
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
# Load Pre-trained ResNet Model
# ================================
print("Loading ResNet model...")
try:
    model_path = "models/resnet_best.h5"
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")

    # Save model summary
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/resnet_model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("✅ Model summary saved!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ================================
# Extract Features with ResNet
# ================================
print("Extracting features using ResNet...")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("global_average_pooling2d").output)

features = []
for img in tqdm(X, desc="Extracting Features"):
    img_expanded = np.expand_dims(img, axis=0)
    feature = feature_extractor.predict(img_expanded, verbose=0)
    features.append(feature)

features = np.array(features).squeeze()
print(f"✅ Extracted Features Shape: {features.shape}")

# ================================
# Apply PCA for Feature Reduction
# ================================
print("Applying PCA for feature reduction...")
pca = PCA(n_components=100)
reduced_features = pca.fit_transform(features)
print(f"✅ PCA Reduced Features Shape: {reduced_features.shape}")

# Save reduced features
np.save("outputs/reduced_features.npy", reduced_features)
print("✅ Reduced features saved!")

# ================================
# Run Disease Detection and Visualization
# ================================
print("Running Disease Detection...")
num_images = 25  # Visualize more images

for i in range(num_images):
    print(f"Processing Image {i + 1}/{num_images}...")
    input_image = np.expand_dims(X[i], axis=0)

    predictions = model.predict(input_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    disease_name = DISEASE_LABELS[predicted_class]

    # Visualization
    plt.imshow(X[i])
    plt.title(f"{disease_name}")
    plt.axis("off")

    # Save visualization
    vis_path = f"outputs/visualization_resnet_{i + 1}.png"
    plt.savefig(vis_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Visualization saved at: {vis_path}")

print("✅ Disease Detection and PCA Complete!")

