# Plant Disease Classification

This project implements a hybrid deep learning model for plant disease classification using both CNN-based deep features and handcrafted features.

## Project Structure
```
.
├── data/
│   └── processed/          # Processed data and features
├── models/                 # Saved model files
├── notebooks/             # Jupyter notebooks for exploration
├── outputs/               # Training outputs and results
├── src/                   # Source code
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   - Place your raw image data in the `data/raw` directory
   - The data should be organized in class-specific folders

2. **Training**:
```bash
python train.py
```
This will:
- Load and preprocess the data
- Extract deep features using InceptionV3
- Extract handcrafted features
- Train the hybrid model
- Save the model in the `models` directory

3. **Evaluation**:
```bash
python evaluate.py
```

## Model Architecture

The project uses a hybrid approach combining:
1. Deep features from InceptionV3
2. Handcrafted features (texture, color, and shape features)
3. A fusion network to combine both feature types

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV
- scikit-learn
- scikit-image

## Memory Optimization

The code includes memory optimization techniques:
- TensorFlow memory growth control
- Chunk-based feature extraction
- Garbage collection during processing

## Acknowledgments
- Dataset source: [Source Name]
- ResNet architecture: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
- Special thanks to [Any specific acknowledgments]

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
Developed with ❤️ by Moksh Arora
