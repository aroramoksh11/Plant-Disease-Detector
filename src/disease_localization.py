import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

# ================================
# Force CPU Execution (Disable Metal)
# ================================
print("Forcing CPU Execution...")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.set_visible_devices([], 'GPU')
print("✅ Forced CPU Execution. Running on CPU only.")

# ================================
# Disease Labels
# ================================
CLASS_LABELS = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Cherry - Powdery Mildew", "Cherry - Healthy", "Corn - Cercospora Leaf Spot", "Corn - Common Rust",
    "Corn - Northern Leaf Blight", "Corn - Healthy", "Grape - Black Rot", "Grape - Esca", "Grape - Leaf Blight",
    "Grape - Healthy", "Peach - Bacterial Spot", "Peach - Healthy", "Potato - Early Blight", "Potato - Late Blight",
    "Potato - Healthy", "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight", "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites", "Tomato - Target Spot", "Tomato - Mosaic Virus",
    "Tomato - Yellow Leaf Curl Virus", "Tomato - Healthy"
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
# Load Pre-trained InceptionV3 Model
# ================================
print("Loading InceptionV3 model...")
try:
    model_path = "models/inception_v3_model.h5"
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")

    # Save model summary
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/inception_model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("✅ Model summary saved!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ================================
# Grad-CAM Implementation (Multi-Layer Fusion)
# ================================
def grad_cam(input_image, model, layer_names):
    heatmaps = []

    for layer_name in layer_names:
        print(f"🔍 Running Grad-CAM on {layer_name}...")

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_image, training=False)
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        # Generate the heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Resize to ensure consistent shape
        if heatmap.shape != (7, 7):  # Adjust according to your model's output size
            heatmap = cv2.resize(heatmap, (7, 7))

        heatmaps.append(heatmap)

    fused_heatmap = np.mean(heatmaps, axis=0)
    
    # Debug: Print predictions
    print(f"Predictions: {predictions.numpy()}")
    print(f"Predicted Class: {predicted_class.numpy()} - {CLASS_LABELS[predicted_class.numpy()]}")
    
    return fused_heatmap, predicted_class.numpy()

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

# Layers to extract heatmaps from (InceptionV3)
layer_names = ["mixed7", "mixed8", "mixed10"]

# Number of images to visualize
num_images = 10  # Visualizing 10 images instead of 5

for i in range(num_images):
    print(f"Processing Image {i + 1}/{num_images}...")

    # Prepare the image for Grad-CAM
    input_image = np.expand_dims(X[i], axis=0)

    # Force the input to run on CPU
    with tf.device('/CPU:0'):
        # Generate Grad-CAM heatmap
        try:
            heatmap, predicted_class = grad_cam(input_image, model, layer_names)

            # Resize heatmap to match input image
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            # Calculate IoU
            iou = calculate_iou(heatmap_resized)
            severity = iou * 100

            # Print predictions
            detected_disease = CLASS_LABELS[predicted_class]
            print(f"🦠 Detected Disease: {detected_disease}")
            print(f"✅ IoU: {iou:.4f}, Severity: {severity:.2f}%")

            # Visualization
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(X[i])
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap="jet")
            plt.title("Heatmap")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(X[i])
            plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.title(f"{detected_disease}\nIoU: {iou:.4f}, Severity: {severity:.2f}%")
            plt.axis("off")

            vis_path = f"outputs/visualization_{i + 1}.png"
            plt.savefig(vis_path)
            plt.close()
            print(f"✅ Visualization saved at: {vis_path}")

        except Exception as e:
            print(f"❌ Error during Grad-CAM/IoU calculation: {e}")

print("✅ Disease Localization and Detection Complete!")
