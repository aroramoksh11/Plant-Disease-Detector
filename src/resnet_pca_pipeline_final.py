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
print("Forcing CPU Execution...")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.set_visible_devices([], 'GPU')
print("✅ Forced CPU Execution. Running on CPU only.")

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

# Disease Labels
disease_labels = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Cherry - Healthy", "Cherry - Powdery Mildew",
    "Corn - Cercospora Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca (Black Measles)", "Grape - Leaf Blight (Isariopsis Leaf Spot)", "Grape - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight", "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites", "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus", "Tomato - Healthy"
]

# ================================
# Load ResNet Model
# ================================
print("Loading ResNet model...")
try:
    model = tf.keras.models.load_model("models/resnet_best.h5")
    print("✅ Model loaded from: models/resnet_best.h5")

    # Save model summary
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("✅ Model summary saved!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ================================
# Extract Features
# ================================
print("Extracting features using ResNet...")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
features = feature_extractor.predict(X, batch_size=32, verbose=1)
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
# Grad-CAM Implementation
# ================================
def grad_cam(input_image, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image, training=False)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap, predicted_class

# ================================
# IoU Calculation
# ================================
def calculate_iou(heatmap, threshold=0.2):
    binary_mask = (heatmap > threshold).astype(np.uint8)
    ground_truth_mask = np.ones_like(binary_mask)

    intersection = np.logical_and(binary_mask, ground_truth_mask).sum()
    union = np.logical_or(binary_mask, ground_truth_mask).sum()

    iou = intersection / (union + 1e-6)
    return iou

# ================================
# Run Grad-CAM and Calculate IoU
# ================================
print("Running Grad-CAM and Disease Detection...")
os.makedirs("outputs", exist_ok=True)

# Visualize 25 random images
np.random.seed(42)
indices = np.random.choice(len(X), 25, replace=False)

for i, idx in enumerate(indices):
    print(f"Processing Image {i + 1}/25...")

    input_image = np.expand_dims(X[idx], axis=0)

    heatmap, predicted_class = grad_cam(input_image, model)
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Normalize heatmap and apply color map
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    input_image_bgr = (X[idx] * 255).astype(np.uint8)
    input_image_bgr = cv2.cvtColor(input_image_bgr, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(input_image_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Calculate IoU and Severity
    iou = calculate_iou(heatmap_resized)
    severity = iou * 100
    severity_level = "Mild" if severity < 30 else "Moderate" if severity < 70 else "Severe"

    # Get disease name
    disease_name = disease_labels[predicted_class]

    # Add text to visualization
    cv2.putText(superimposed_img, f"Disease: {disease_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(superimposed_img, f"IoU: {iou:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(superimposed_img, f"Severity: {severity:.2f}% ({severity_level})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save visualization
    vis_path = f"outputs/visualization_resnet_{i + 1}.png"
    cv2.imwrite(vis_path, superimposed_img)
    print(f"✅ Visualization saved at: {vis_path}")

print("✅ Disease Localization, Feature Extraction, and PCA Complete!")
