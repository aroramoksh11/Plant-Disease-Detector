import numpy as np
import tensorflow as tf
import cv2
import os

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy(), predicted_class

def overlay_heatmap(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_normalized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    return superimposed_img

if __name__ == "__main__":
    model = tf.keras.applications.ResNet50(weights="imagenet")  # Load Model
    image_path = "sample_image.jpg"
    img_array = preprocess_image(image_path)
    heatmap, _ = grad_cam(model, img_array, "conv5_block3_out")
    
    original_img = cv2.imread(image_path)
    output_img = overlay_heatmap(original_img, heatmap)
    cv2.imwrite("gradcam_output.png", output_img)
    print("✅ Grad-CAM saved as 'gradcam_output.png'")
