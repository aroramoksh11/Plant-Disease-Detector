import numpy as np
import tensorflow as tf
import cv2
import os
import logging

# ================================
# Setup Logging
# ================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================================
# Hardcoded Disease Labels
# ================================
DISEASE_LABELS = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca", "Grape - Leaf Blight", "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot", "Peach - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites", "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", "Tomato - Healthy"
]

# ================================
# Load Data
# ================================
def load_data(X_path="data/processed/X.npy", y_path="data/processed/y.npy"):
    logging.info("🔄 Loading dataset...")
    try:
        X = np.load(X_path)
        y = np.load(y_path)
        logging.info(f"✅ Data successfully loaded! X Shape: {X.shape}, y Shape: {y.shape}")
        return X, y
    except FileNotFoundError as e:
        logging.error(f"❌ File not found: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error loading data: {e}")
        exit(1)

# ================================
# Load Pre-trained ResNet Model
# ================================
def load_model(model_path="models/resnet_best.h5"):
    logging.info("🔄 Loading ResNet model...")
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"✅ Model successfully loaded from: {model_path}")
        return model
    except FileNotFoundError as e:
        logging.error(f"❌ Model file not found: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error loading model: {e}")
        exit(1)

# ================================
# Grad-CAM Implementation
# ================================
def grad_cam(input_image, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        inputs=[model.input], 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image, training=False)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)  # Avoid division by zero
    return heatmap.numpy(), predicted_class.numpy()

# ================================
# IoU Calculation
# ================================
def calculate_iou(heatmap, threshold=0.2):
    binary_mask = (heatmap > threshold).astype(np.uint8)
    ground_truth_mask = np.ones_like(binary_mask)  # Assume lesion covers full leaf

    intersection = np.logical_and(binary_mask, ground_truth_mask).sum()
    union = np.logical_or(binary_mask, ground_truth_mask).sum()

    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou

# ================================
# Visualize and Save Results
# ================================
def visualize_and_save(X, indices, model, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        logging.info(f"Processing Image {i + 1}/{len(indices)}...")

        input_image = np.expand_dims(X[idx], axis=0)

        # Generate Grad-CAM heatmap
        heatmap, predicted_class = grad_cam(input_image, model)
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        # Calculate IoU and Severity
        iou = calculate_iou(heatmap_resized)
        severity = iou * 100
        severity_level = "Mild" if severity < 30 else "Moderate" if severity < 70 else "Severe"

        # Get disease name
        disease_name = DISEASE_LABELS[predicted_class]

        # Normalize heatmap and apply color map
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Superimpose heatmap on original image
        input_image_bgr = (X[idx] * 255).astype(np.uint8)
        input_image_bgr = cv2.cvtColor(input_image_bgr, cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(input_image_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Create side-by-side visualization
        combined_img = np.hstack([input_image_bgr, superimposed_img])

        # Create a black box below the image
        black_box = np.zeros((50, combined_img.shape[1], 3), dtype=np.uint8)
        cv2.putText(black_box, f"Disease: {disease_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(black_box, f"IoU: {iou:.4f}  Severity: {severity:.2f}% ({severity_level})", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Concatenate the combined image with the black box
        final_output = np.vstack([combined_img, black_box])

        # Save visualization
        vis_path = os.path.join(output_dir, f"visualization_resnet_iou_{i + 1}.png")
        cv2.imwrite(vis_path, final_output)
        logging.info(f"✅ Visualization saved at: {vis_path}")

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    X, y = load_data()
    model = load_model()

    # Randomly select 25 images for visualization
    np.random.seed(42)
    selected_indices = np.random.choice(len(X), 25, replace=False)

    visualize_and_save(X, selected_indices, model)

    logging.info("✅ Disease Localization, Grad-CAM, IoU, and Severity Detection Complete!")
