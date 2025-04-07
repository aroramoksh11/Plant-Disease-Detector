"""
Enhanced training script for plant disease classification with detailed outputs and comprehensive heatmap analysis.
"""

import os
import numpy as np
import tensorflow as tf
import logging
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from src.data_processing import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.visualization import Visualizer
from src.pca_analysis import PCAAnalyzer
from src.resnet_model import ResNetModel
from src.ml_classifiers import MLClassifiers
from src.heatmap import HeatmapGenerator
from src.disease_classifier import DiseaseClassifier
from src.iou_calculator import IoUCalculator
from src.config import LOGGING_CONFIG, MODEL_CONFIG, ML_CONFIG, VIS_CONFIG, OUTPUTS_DIR

def setup_logging():
    """Setup enhanced logging with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    LOGGING_CONFIG["handlers"]["file"]["filename"] = str(log_dir / f"training_{timestamp}.log")
    logging.basicConfig(**LOGGING_CONFIG)
    return logging.getLogger(__name__)

def print_section_header(logger, title):
    """Print formatted section header."""
    logger.info("\n" + "=" * 50)
    logger.info(f" {title} ")
    logger.info("=" * 50)

def save_layer_activations(heatmap_generator, img_batch, save_dir):
    """Save activation patterns for each convolutional layer."""
    activation_stats = heatmap_generator.analyze_activation_patterns(img_batch)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activation statistics
    with open(save_dir / "activation_stats.txt", "w") as f:
        for layer_name, stats in activation_stats.items():
            f.write(f"\nLayer: {layer_name}\n")
            f.write("-" * 30 + "\n")
            for stat_name, value in stats.items():
                if stat_name != 'shape':
                    f.write(f"{stat_name}: {value}\n")
                else:
                    f.write(f"{stat_name}: {list(value)}\n")

def generate_comparative_heatmaps(heatmap_generator, img_batch, predictions, class_names, save_dir):
    """Generate comparative heatmaps for correct and incorrect predictions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    true_labels = np.argmax(img_batch[1], axis=1)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Generate heatmaps for both correct and incorrect predictions
    for i, (img, true_label, pred_label) in enumerate(zip(img_batch[0], true_labels, pred_labels)):
        is_correct = true_label == pred_label
        status = "correct" if is_correct else "incorrect"
        
        # Generate heatmap
        heatmap = heatmap_generator.generate_grad_cam(
            img[np.newaxis, ...],
            class_idx=pred_label
        )
        
        # Save visualization with detailed information
        save_path = save_dir / f"heatmap_{status}_{i}.png"
        heatmap_generator.visualize_heatmap(
            img,
            heatmap,
            save_path=save_path,
            show_original=True,
            title=f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
        )

def main():
    """Enhanced main training function with detailed outputs and heatmap analysis."""
    logger = setup_logging()
    start_time = time.time()
    
    try:
        print_section_header(logger, "PLANT DISEASE CLASSIFICATION TRAINING")
        
        # System information
        logger.info("\nSystem Information:")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
        logger.info(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize components with progress tracking
        print_section_header(logger, "INITIALIZING COMPONENTS")
        components = {
            "Data Processor": DataProcessor(),
            "Feature Extractor": FeatureExtractor(),
            "Visualizer": Visualizer(),
            "PCA Analyzer": PCAAnalyzer(),
            "Disease Classifier": DiseaseClassifier(),
            "IoU Calculator": IoUCalculator(),
            "ML Classifiers": MLClassifiers(random_state=ML_CONFIG["random_forest"]["random_state"])
        }
        
        for name, component in components.items():
            logger.info(f"✓ Initialized {name}")
        
        # Data loading and preprocessing with progress bar
        print_section_header(logger, "DATA LOADING AND PREPROCESSING")
        with tqdm(total=3, desc="Loading data") as pbar:
            logger.info("Loading training data...")
            train_data, val_data, test_data = components["Data Processor"].load_and_preprocess_data("data/raw")
            pbar.update(3)
        
        # Print dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        
        # Feature extraction with progress tracking
        print_section_header(logger, "FEATURE EXTRACTION")
        with tqdm(total=3, desc="Extracting features") as pbar:
            logger.info("Extracting features from training data...")
            X_train, y_train = components["Feature Extractor"].extract_all_features(train_data)
            pbar.update(1)
            
            logger.info("Extracting features from validation data...")
            X_val, y_val = components["Feature Extractor"].extract_all_features(val_data)
            pbar.update(1)
            
            logger.info("Extracting features from test data...")
            X_test, y_test = components["Feature Extractor"].extract_all_features(test_data)
            pbar.update(1)
        
        # PCA Analysis with detailed output
        print_section_header(logger, "PCA ANALYSIS")
        X_train_pca = components["PCA Analyzer"].fit_transform(X_train)
        logger.info("PCA Analysis Results:")
        logger.info(f"Explained variance ratio: {components['PCA Analyzer'].pca.explained_variance_ratio_[:5]}")
        
        # ML Classifiers Training with progress tracking
        print_section_header(logger, "TRAINING ML CLASSIFIERS")
        ml_classifiers = components["ML Classifiers"]
        with tqdm(total=4, desc="Training classifiers") as pbar:
            logger.info("Training Random Forest...")
            ml_classifiers.train_random_forest(X_train, y_train)
            pbar.update(1)
            
            logger.info("Training SVM...")
            ml_classifiers.train_svm(X_train, y_train)
            pbar.update(1)
            
            logger.info("Training Logistic Regression...")
            ml_classifiers.train_logistic_regression(X_train, y_train)
            pbar.update(1)
            
            logger.info("Training Voting Classifier...")
            ml_classifiers.train_voting_classifier(X_train, y_train)
            pbar.update(1)
        
        # Evaluate ML classifiers
        ml_results = ml_classifiers.evaluate(X_test, y_test, components["Data Processor"].class_names)
        
        # ResNet Training with progress tracking
        print_section_header(logger, "TRAINING RESNET MODEL")
        resnet_model = ResNetModel(num_classes=len(components["Data Processor"].class_names))
        resnet_model.compile()
        
        # Create data generators
        train_generator = components["Data Processor"].create_data_generator(train_data, is_training=True)
        val_generator = components["Data Processor"].create_data_generator(val_data, is_training=False)
        
        # Initialize HeatmapGenerator
        heatmap_generator = HeatmapGenerator(resnet_model.model)
        
        # Training callback for activation analysis
        class ActivationAnalysisCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0:  # Every 5 epochs
                    batch_images, _ = next(val_generator)
                    save_dir = OUTPUTS_DIR / "activations" / f"epoch_{epoch+1}"
                    save_layer_activations(heatmap_generator, batch_images, save_dir)
        
        # Train ResNet with progress tracking and activation analysis
        history = resnet_model.train(
            train_generator,
            val_generator,
            epochs=MODEL_CONFIG["epochs"],
            callbacks=[
                tf.keras.callbacks.ProgbarLogger(count_mode='steps'),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(OUTPUTS_DIR / "models" / "resnet_best.h5"),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                ActivationAnalysisCallback()
            ]
        )
        
        # Generate visualizations
        print_section_header(logger, "GENERATING VISUALIZATIONS")
        
        # Training history plots
        components["Visualizer"].plot_training_history(history)
        logger.info("✓ Saved training history plots")
        
        # Evaluate models
        print_section_header(logger, "MODEL EVALUATION")
        test_generator = components["Data Processor"].create_data_generator(test_data, is_training=False)
        resnet_metrics = resnet_model.evaluate(test_generator)
        
        # Generate comprehensive heatmap analysis
        print_section_header(logger, "GENERATING HEATMAP ANALYSIS")
        
        # Get a batch of test images
        test_batch = next(test_generator)
        predictions = resnet_model.model.predict(test_batch[0])
        
        # Generate comparative heatmaps
        generate_comparative_heatmaps(
            heatmap_generator,
            test_batch,
            predictions,
            components["Data Processor"].class_names,
            OUTPUTS_DIR / "visualizations" / "heatmaps" / "comparative"
        )
        
        # Save layer activations for final model
        save_layer_activations(
            heatmap_generator,
            test_batch[0],
            OUTPUTS_DIR / "activations" / "final_model"
        )
        
        # Generate batch heatmaps with progress tracking
        with tqdm(total=len(test_batch[0]), desc="Generating batch heatmaps") as pbar:
            heatmaps = heatmap_generator.generate_multiple_heatmaps(
                test_batch[0],
                class_indices=np.argmax(predictions, axis=1),
                save_dir=OUTPUTS_DIR / "visualizations" / "heatmaps" / "batch",
                batch_size=8
            )
            pbar.update(len(test_batch[0]))
        
        # Disease Analysis with progress tracking
        print_section_header(logger, "DISEASE ANALYSIS")
        
        with tqdm(total=len(test_batch[0]), desc="Analyzing diseases") as pbar:
            for i, (img, pred) in enumerate(zip(test_batch[0], predictions)):
                # Get disease mask and heatmap
                mask = components["Feature Extractor"].segment_disease(img)
                heatmap = heatmaps[i]
                
                # Calculate IoU between heatmap and disease mask
                iou = components["IoU Calculator"].calculate_iou(
                    (heatmap > 0.5).astype(np.uint8),
                    mask
                )
                
                # Generate and save analysis
                report = components["Disease Classifier"].generate_report(
                    img, pred, mask, heatmap_iou=iou
                )
                
                components["Disease Classifier"].visualize_analysis(
                    img,
                    mask,
                    report,
                    heatmap=heatmap,
                    save_path=OUTPUTS_DIR / "visualizations" / "disease_analysis" / f"analysis_{i}.png"
                )
                pbar.update(1)
        
        # Save models
        print_section_header(logger, "SAVING MODELS")
        ml_classifiers.save_models(OUTPUTS_DIR / "models")
        resnet_model.save_model(OUTPUTS_DIR / "models" / "resnet_model.h5")
        logger.info("✓ Saved all models")
        
        # Generate final report
        print_section_header(logger, "GENERATING FINAL REPORT")
        generate_comprehensive_report(
            OUTPUTS_DIR / "reports",
            ml_results,
            resnet_metrics,
            components["Data Processor"].class_names,
            start_time,
            heatmap_analysis=True
        )
        
        # Training complete
        end_time = time.time()
        training_time = end_time - start_time
        print_section_header(logger, "TRAINING COMPLETE")
        logger.info(f"Total training time: {training_time/3600:.2f} hours")
        logger.info(f"Results saved in: {OUTPUTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def generate_comprehensive_report(output_dir, ml_results, resnet_metrics, class_names, start_time, heatmap_analysis=False):
    """Generate detailed training report with all metrics and results."""
    report_path = output_dir / "training_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("PLANT DISEASE CLASSIFICATION - TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Training information
        f.write("Training Information:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {(time.time() - start_time)/3600:.2f} hours\n\n")
        
        # Model configurations
        f.write("Model Configurations:\n")
        f.write("-" * 50 + "\n")
        f.write("ResNet Configuration:\n")
        for key, value in MODEL_CONFIG["resnet"].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nML Classifiers Configuration:\n")
        for model, config in ML_CONFIG.items():
            f.write(f"\n{model}:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
        
        # Performance metrics
        f.write("\nPerformance Metrics:\n")
        f.write("-" * 50 + "\n")
        
        # ML Classifiers performance
        f.write("\n1. ML Classifiers Performance:\n")
        for model_name, report in ml_results.items():
            f.write(f"\n{model_name.replace('_', ' ').title()}:\n")
            f.write(f"Overall Accuracy: {report['accuracy']:.4f}\n")
            f.write("Per-class metrics:\n")
            for class_name in class_names:
                metrics = report[class_name]
                f.write(f"\n  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1-score: {metrics['f1-score']:.4f}\n")
        
        # ResNet performance
        f.write("\n2. ResNet Model Performance:\n")
        f.write(f"Test Loss: {resnet_metrics[0]:.4f}\n")
        f.write(f"Test Accuracy: {resnet_metrics[1]:.4f}\n")
        
        # Generated files
        f.write("\nGenerated Files:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Model Files:\n")
        f.write("   - resnet_model.h5\n")
        f.write("   - random_forest_model.joblib\n")
        f.write("   - svm_model.joblib\n")
        f.write("   - logistic_regression_model.joblib\n")
        
        f.write("\n2. Visualization Files:\n")
        f.write("   - Training history plots\n")
        f.write("   - Confusion matrices\n")
        f.write("   - Grad-CAM heatmaps\n")
        f.write("   - Disease analysis visualizations\n")
        
        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) available: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU memory growth error: {str(e)}")
    
    main()
