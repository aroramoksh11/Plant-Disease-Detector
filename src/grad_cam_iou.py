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
# Load Pre-trained CNN Model
# ================================
print("Loading CNN model...")
try:
    model = tf.keras.models.load_model("models/inception_v3_model.h5")
    print("✅ Model loaded!")

    # Save model summary
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("✅ Model summary saved!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)


# ================================
# Grad-CAM Implementation
# ================================
def grad_cam(input_image, model, layer_name="mixed10"):
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
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    heatmap = heatmap.numpy()

    return heatmap, predicted_class


# ================================
# IoU Calculation
# ================================
def calculate_iou(heatmap, threshold=0.2):  # Dynamic thresholding
    # Binarize the heatmap using the threshold
    binary_mask = (heatmap > threshold).astype(np.uint8)
    ground_truth_mask = np.ones_like(binary_mask)  # Simulated ground truth (for demo)

    intersection = np.logical_and(binary_mask, ground_truth_mask).sum()
    union = np.logical_or(binary_mask, ground_truth_mask).sum()

    iou = intersection / (union + 1e-6)
    return iou


# ================================
# Run Grad-CAM and Calculate IoU
# ================================
print("Running Grad-CAM and IoU Calculation...")
os.makedirs("outputs", exist_ok=True)

# Limit the number of images for demo purposes
num_images = 5
layers_to_try = ["mixed7", "mixed8", "mixed10"]

iou_scores = {}

for i in range(num_images):
    print(f"Processing Image {i + 1}/{num_images}...")

    # Prepare the image for Grad-CAM
    input_image = np.expand_dims(X[i], axis=0)

    iou_scores[i] = {}

    for layer_name in layers_to_try:
        print(f"🔍 Running Grad-CAM on {layer_name}...")

        # Generate Grad-CAM heatmap
        heatmap, predicted_class = grad_cam(input_image, model, layer_name)

        # Resize heatmap to match input image
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        # Dynamic threshold based on max intensity
        threshold = 0.2 * np.max(heatmap_resized)

        # Normalize heatmap and apply color map
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Convert input image to uint8 and BGR for OpenCV
        input_image_bgr = (X[i] * 255).astype(np.uint8)
        input_image_bgr = cv2.cvtColor(input_image_bgr, cv2.COLOR_RGB2BGR)

        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(input_image_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Save Grad-CAM result
        output_dir = f"outputs/image_{i + 1}"
        os.makedirs(output_dir, exist_ok=True)
        grad_cam_path = os.path.join(output_dir, f"grad_cam_{layer_name}.png")
        cv2.imwrite(grad_cam_path, superimposed_img)
        print(f"✅ Grad-CAM saved at: {grad_cam_path}")

        # Calculate IoU
        iou = calculate_iou(heatmap_resized, threshold)
        iou_scores[i][layer_name] = iou
        print(f"✅ Image {i + 1} - {layer_name} IoU: {iou:.4f}")

        # Plot and save IoU visualization
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(X[i])
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(heatmap, cmap="jet")
        ax[1].set_title(f"Heatmap - {layer_name}")
        ax[1].axis("off")

        ax[2].imshow(superimposed_img[:, :, ::-1])  # Convert BGR to RGB
        ax[2].set_title(f"Superimposed - IoU: {iou:.4f}")
        ax[2].axis("off")

        iou_vis_path = os.path.join(output_dir, f"iou_visualization_{layer_name}.png")
        plt.tight_layout()
        plt.savefig(iou_vis_path)
        plt.close()
        print(f"✅ IoU visualization saved at: {iou_vis_path}")

# Save IoU scores
np.save("outputs/iou_scores.npy", iou_scores)
print("✅ IoU scores saved!")

print("✅ Grad-CAM and IoU calculation complete!")
