# AgriVision AI: Advanced Plant Disease Detection System

An advanced deep learning project for automated plant disease classification using computer vision and machine learning techniques. This project combines ResNet architecture with traditional ML classifiers to provide robust plant disease detection and analysis.

## Features

- Advanced plant disease classification using ResNet architecture
- Multiple ML classifier implementations (Random Forest, SVM, Logistic Regression)
- Comprehensive visualization tools including Grad-CAM heatmaps
- Detailed performance metrics and analysis
- Automated data processing and augmentation
- Training progress tracking and logging
- Extensive evaluation metrics and reporting
- PCA analysis for feature dimensionality reduction
- IoU (Intersection over Union) calculation for model evaluation
- Disease-specific analysis and visualization

## Requirements

- Python 3.8+
- macOS with Apple Silicon support (M1/M2 chips)
- GPU acceleration support
- Required Python packages listed in `requirements.txt`
- Minimum 16GB RAM recommended
- 50GB+ free disk space for dataset and model storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd plant-disease-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Verify the setup:
```bash
python verify_setup.py
```

## Project Structure

```
.
├── data/               # Data directory for training/validation/test sets
│   ├── raw/           # Raw input images
│   └── processed/     # Processed and augmented data
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks for analysis
├── outputs/           # Training outputs and visualizations
│   ├── logs/         # Training and execution logs
│   ├── heatmaps/     # Generated Grad-CAM heatmaps
│   └── reports/      # Performance reports and metrics
├── src/              # Source code
│   ├── config.py           # Configuration settings
│   ├── data_processing.py  # Data preprocessing utilities
│   ├── feature_extractor.py # Feature extraction modules
│   ├── resnet_model.py     # ResNet model implementation
│   ├── ml_classifiers.py   # ML classifier implementations
│   ├── visualization.py    # Visualization utilities
│   ├── training_metrics.py # Training metrics tracking
│   ├── heatmap.py         # Grad-CAM heatmap generation
│   ├── pca_analysis.py    # PCA analysis utilities
│   ├── iou_calculator.py  # IoU calculation utilities
│   ├── disease_classifier.py # Disease classification logic
│   ├── model_enhancements.py # Model enhancement utilities
│   └── model_builder.py    # Model building utilities
├── clean_directories.py    # Directory cleanup utilities
├── evaluate.py            # Model evaluation script
├── main.py               # Main execution script
├── organize_data.py      # Data organization utilities
├── requirements.txt      # Project dependencies
├── run.sh               # Shell script for running the pipeline
├── setup_data.py        # Data setup script
├── train.py             # Training script
└── validate_setup.py    # Setup validation script
```

## Usage

1. Prepare your dataset:
```bash
python setup_data.py
```

2. Organize the data:
```bash
python organize_data.py
```

3. Train the model:
```bash
python train.py
```

4. Evaluate the model:
```bash
python evaluate.py
```

Alternatively, you can run the entire pipeline using:
```bash
./run.sh
```

## Model Architecture

The project implements a hybrid approach combining:
- ResNet-based deep learning model for feature extraction
- Multiple ML classifiers for comparison and ensemble predictions
- Custom layers and enhancements for improved performance
- Feature extraction pipeline with multiple techniques
- PCA-based dimensionality reduction
- IoU-based evaluation metrics

## Features and Capabilities

### 1. Data Processing
- Automated data augmentation
- Preprocessing pipeline
- Dataset splitting and validation
- Image resizing and normalization
- Data augmentation techniques
- Batch processing support

### 2. Training
- Multi-model training support
- Progress tracking and checkpointing
- Early stopping and learning rate scheduling
- Cross-validation support
- Model checkpointing
- Training metrics visualization

### 3. Visualization
- Training progress plots
- Confusion matrices
- Grad-CAM heatmaps for model interpretability
- Activation pattern analysis
- Feature importance visualization
- Disease-specific visualizations

### 4. Evaluation
- Comprehensive metrics calculation
- Per-class performance analysis
- Comparative model evaluation
- Detailed reporting
- IoU-based evaluation
- Cross-validation results

## Output and Results

The system generates:
- Trained model files
- Performance metrics and visualizations
- Detailed training reports
- Activation analysis
- Comparative heatmaps
- Classification reports
- PCA analysis results
- Feature importance plots
- Disease-specific analysis reports

All outputs are organized in the `outputs/` directory with timestamp-based subdirectories.

## Configuration

The project uses configuration files in the `src/` directory:
- Model configurations
- Training parameters
- Visualization settings
- Logging configurations
- Data processing settings
- Feature extraction parameters
- Evaluation metrics settings

### Key Configuration Parameters
- Image size: 224x224
- Batch size: 32
- Validation split: 20%
- Test split: 10%
- Random seed: 42
- Learning rate: Configurable
- Early stopping patience: Configurable

## Logging

Comprehensive logging is implemented with:
- Training progress logs
- Error tracking
- Performance metrics
- System diagnostics
- Model checkpoints
- Validation results
- Test results
- Feature extraction logs

Logs are stored in the `logs/` directory with timestamp-based organization.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- scikit-learn team for ML implementations
- The research community for plant disease datasets
- ResNet paper authors for the architecture
- Grad-CAM paper authors for the visualization technique

---
Developed by Moksh Arora
