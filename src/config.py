"""
Configuration settings for the plant disease classification project.
"""

from pathlib import Path

# Directory paths
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
    "class_names": [
        "bacterial_blight",
        "blast",
        "brown_spot",
        "healthy"
    ]
}

# Model configuration
MODEL_CONFIG = {
    "epochs": 50,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.1,
    "resnet": {
        "weights": "imagenet",
        "include_top": False,
        "input_shape": (224, 224, 3),
        "dense_units": 1024,
        "dropout_rate": 0.5
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    }
}

# ML Classifiers configuration
ML_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": 42
    },
    "logistic_regression": {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    },
    "voting": {
        "voting": "soft",
        "weights": [2, 1, 1]  # Weights for RF, SVM, and LR respectively
    }
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "color_features": True,
    "texture_features": True,
    "shape_features": True,
    "hog_features": True,
    "lbp_features": True,
    "pca_components": 50
}

# Visualization configuration
VIS_CONFIG = {
    "plot_size": (12, 8),
    "dpi": 300,
    "cmap": "viridis",
    "font_size": 12,
    "title_font_size": 14,
    "label_font_size": 12,
    "tick_font_size": 10
}

# Disease severity configuration
SEVERITY_CONFIG = {
    "low_threshold": 0.2,
    "medium_threshold": 0.4,
    "high_threshold": 0.6,
    "severity_colors": {
        "low": (0, 255, 0),    # Green
        "medium": (255, 255, 0),  # Yellow
        "high": (255, 0, 0)    # Red
    }
}

# IoU configuration
IOU_CONFIG = {
    "threshold": 0.5,
    "visualization_colors": {
        "true_positive": (0, 255, 0),    # Green
        "false_positive": (255, 0, 0),   # Red
        "false_negative": (0, 0, 255)    # Blue
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "plant_disease.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
} 