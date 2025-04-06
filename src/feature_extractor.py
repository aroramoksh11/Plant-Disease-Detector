import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import gc


def extract_handcrafted_features(img):
    print("🔍 Extracting handcrafted features for one image...")

    # ================================
    # Fix: Handle Batch Input (Take 1 Image)
    # ================================
    if len(img.shape) == 4:
        print("⚠️ Detected batch of images. Taking the first image.")
        img = img[0]  # Take the first image in the batch

    # ================================
    # Step 1: Extract Color Features
    # ================================
    print("Step 1: Extracting color features...")

    # Ensure the image is uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Calculate means and stds directly using NumPy
    means = np.mean(img, axis=(0, 1))
    stds = np.std(img, axis=(0, 1))
    color_features = np.hstack([means, stds])
    print("✅ Color features extracted:", color_features)

    # ================================
    # Step 2: Extract Texture Features (Fix)
    # ================================
    print("Step 2: Extracting texture features...")

    # Check image shape to avoid OpenCV error
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert RGB to Grayscale only if it has 3 channels
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        print("⚠️ Image is already grayscale.")
        gray = img

    # Extract GLCM texture features
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    texture_features = [contrast, correlation]
    print("✅ Texture features extracted:", texture_features)

    # ================================
    # Step 3: Combine Features
    # ================================
    print("Step 3: Combining features...")
    features = np.hstack([color_features, texture_features])
    print("✅ Features combined:", features)

    # Cleanup after each image to prevent memory leaks
    del gray, glcm
    gc.collect()

    print("✅ Handcrafted features extracted for one image.")
    return features
