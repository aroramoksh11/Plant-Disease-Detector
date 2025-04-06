import os
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

# ============================
# Config
# ============================
OUTPUT_DIR = "outputs"
REPORT_PATH = os.path.join(OUTPUT_DIR, "grad_cam_report.pdf")
IOU_PLOT_PATH = os.path.join(OUTPUT_DIR, "iou_plot.png")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 1. Plot IoU Scores 📊
# ============================
def plot_iou_scores(iou_scores):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(iou_scores) + 1), iou_scores, color="royalblue")
    plt.xticks(range(1, len(iou_scores) + 1))
    plt.xlabel("Image Index")
    plt.ylabel("IoU Score")
    plt.title("IoU Scores for Grad-CAM")
    plt.savefig(IOU_PLOT_PATH)
    plt.close()
    print(f"✅ IoU Plot saved at: {IOU_PLOT_PATH}")

# ============================
# 2. Generate PDF Report 📈
# ============================
def generate_report(iou_scores, model_summary_path="outputs/model_summary.txt"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Plant Disease Classification - Grad-CAM Analysis", ln=True, align="C")
    pdf.ln(10)

    # Model Summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Model Summary:", ln=True)
    pdf.set_font("Arial", "", 10)
    with open(model_summary_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            pdf.multi_cell(0, 6, line.strip())
    pdf.ln(10)

    # IoU Plot
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "IoU Scores:", ln=True)
    pdf.image(IOU_PLOT_PATH, x=10, w=180)
    pdf.ln(10)

    # Grad-CAM Images and IoU Scores
    pdf.cell(0, 10, "Grad-CAM Results:", ln=True)
    for i, iou in enumerate(iou_scores, 1):
        grad_cam_path = os.path.join(OUTPUT_DIR, f"grad_cam_{i}.png")
        pdf.image(grad_cam_path, x=20, w=160)
        pdf.cell(0, 10, f"Image {i} - IoU: {iou:.4f}", ln=True)
        pdf.ln(5)

    # Save PDF
    pdf.output(REPORT_PATH)
    print(f"✅ Report generated at: {REPORT_PATH}")

# ============================
# 3. Compare Inception-V3 vs ResNet 🧠
# ============================
def compare_grad_cams(n_images=5):
    comparison_path = os.path.join(OUTPUT_DIR, "grad_cam_comparison.png")
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 2 * n_images))

    for i in range(1, n_images + 1):
        # Inception-V3
        inception_path = os.path.join(OUTPUT_DIR, f"grad_cam_inception_{i}.png")
        resnet_path = os.path.join(OUTPUT_DIR, f"grad_cam_resnet_{i}.png")

        if os.path.exists(inception_path) and os.path.exists(resnet_path):
            inception_img = Image.open(inception_path)
            resnet_img = Image.open(resnet_path)

            axes[i - 1, 0].imshow(inception_img)
            axes[i - 1, 0].axis("off")
            axes[i - 1, 0].set_title(f"Inception-V3 - Image {i}")

            axes[i - 1, 1].imshow(resnet_img)
            axes[i - 1, 1].axis("off")
            axes[i - 1, 1].set_title(f"ResNet - Image {i}")
        else:
            print(f"⚠️ Missing Grad-CAM images for Image {i}")

    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    print(f"✅ Grad-CAM Comparison saved at: {comparison_path}")


# ============================
# Run all analyses 🚀
# ============================
if __name__ == "__main__":
    # Load IoU scores
    iou_scores = np.load(os.path.join(OUTPUT_DIR, "iou_scores.npy"))
    
    print("📊 Plotting IoU Scores...")
    plot_iou_scores(iou_scores)

    print("📈 Generating PDF Report...")
    generate_report(iou_scores)

    print("🧠 Comparing Grad-CAMs...")
    compare_grad_cams()

    print("✅ All analyses complete!")
