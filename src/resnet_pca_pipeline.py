import numpy as np
import tensorflow as tf
import cv2
import os
from sklearn.decomposition import PCA

# ================================
# Force CPU Execution (Disable Metal)
# ================================
print("Forcing CPU Execution...")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.set_visible_devices([], 'GPU')  # Ensures only CPU is used
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

# ================================
# Load ResNet Model
# ================================
print("Loading ResNet model...")
try:
    resnet_model = tf.keras.models.load_model("models/resnet_best.h5")
    print("✅ Model loaded from: models/resnet_best.h5")

    # Save model summary
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/resnet_model_summary.txt", "w") as f:
        resnet_model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("✅ Model summary saved!")

except Exception as e:
    print(f"❌ Error loading ResNet model: {e}")
    exit(1)

# ================================
# Extract ResNet Features
# ================================
def extract_features(model, X, layer_name):
    feature_extractor = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    features = feature_extractor.predict(X, batch_size=32, verbose=1)
    print(f"✅ Extracted Features Shape: {features.shape}")
    return features

print("Extracting features using ResNet...")
resnet_features = extract_features(resnet_model, X, "global_average_pooling2d")

# ================================
# Apply PCA for Feature Reduction
# ================================
print("Applying PCA for feature reduction...")
pca = PCA(n_components=100)  # Reduce to 100 features
reduced_features = pca.fit_transform(resnet_features)
print(f"✅ PCA Reduced Features Shape: {reduced_features.shape}")

# Save reduced features
np.save("data/processed/reduced_features.npy", reduced_features)
print("✅ Reduced features saved!")

# ================================
# Grad-CAM Implementation (ResNet)
# ================================
def grad_cam(input_image, model, layer_name):
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

    return heatmap, predicted_class

# ================================
# IoU Calculation
# ================================
def calculate_iou(heatmap, threshold=0.2):  # Lowered threshold for better detection
    binary_mask = (heatmap > threshold).astype(np.uint8)
    ground_truth_mask = np.ones_like(binary_mask)  # Simulated ground truth (for demo)

    intersection = np.logical_and(binary_mask, ground_truth_mask).sum()
    union = np.logical_or(binary_mask, ground_truth_mask).sum()

    iou = intersection / (union + 1e-6)
    return iou

# ================================
# Run Grad-CAM and Calculate IoU
# ================================
print("Running Grad-CAM and Disease Detection...")
os.makedirs("outputs", exist_ok=True)

# Limit the number of images for demo purposes
num_images = 5
layer_name = "conv5_block3_out"  # Use the last convolutional layer in ResNet

for i in range(num_images):
    print(f"Processing Image {i + 1}/{num_images}...")

    # Prepare the image for Grad-CAM
    input_image = np.expand_dims(X[i], axis=0)

    # Force the input to run on CPU
    with tf.device('/CPU:0'):
        # Generate Grad-CAM heatmap
        try:
            heatmap, predicted_class = grad_cam(input_image, resnet_model, layer_name)

            # Resize heatmap to match input image
            heatmap_resized = cv2.resize(heatmap, (224, 224))

            # Normalize heatmap and apply color map
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            # Convert input image to uint8 and BGR for OpenCV
            input_image_bgr = (X[i] * 255).astype(np.uint8)
            input_image_bgr = cv2.cvtColor(input_image_bgr, cv2.COLOR_RGB2BGR)

            # Superimpose heatmap on original image
            superimposed_img = cv2.addWeighted(input_image_bgr, 0.6, heatmap_colored, 0.4, 0)

            # Calculate IoU
            iou = calculate_iou(heatmap_resized)
            print(f"✅ Image {i + 1} - IoU: {iou:.4f}")

            # Calculate Disease Severity (based on IoU)
            disease_severity = iou * 100
            severity_label = (
                "Mild" if disease_severity < 30 else
                "Moderate" if disease_severity < 70 else
                "Severe"
            )
            print(f"🌡️ Disease Severity: {disease_severity:.2f}% ({severity_label})")

            # Save visualization
            vis_path = f"outputs/visualization_resnet_{i + 1}.png"
            cv2.putText(superimposed_img, f"IoU: {iou:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(superimposed_img, f"Severity: {disease_severity:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(vis_path, superimposed_img)
            print(f"✅ Visualization saved at: {vis_path}")

        except Exception as e:
            print(f"❌ Error during Grad-CAM/IoU calculation: {e}")

print("✅ Disease Localization, Feature Extraction, and PCA Complete!")
