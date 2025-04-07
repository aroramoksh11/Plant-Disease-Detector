"""
Machine learning classifiers module for plant disease classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from .config import VIS_CONFIG, OUTPUTS_DIR, LOGGING_CONFIG

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class MLClassifiers:
    def __init__(self, random_state=42):
        """Initialize machine learning classifiers."""
        logger.info("Initializing ML classifiers")
        
        # Initialize individual classifiers
        self.rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state
        )
        
        self.svm_clf = SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state
        )
        
        self.lr_clf = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=random_state
        )
        
        # Initialize voting classifier
        self.voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.rf_clf),
                ('svm', self.svm_clf),
                ('lr', self.lr_clf)
            ],
            voting='soft'
        )
        
        self.models = {
            'random_forest': self.rf_clf,
            'svm': self.svm_clf,
            'logistic_regression': self.lr_clf,
            'voting': self.voting_clf
        }
        
    def train(self, X_train, y_train):
        """Train all classifiers."""
        logger.info("Training ML classifiers")
        
        for name, model in self.models.items():
            logger.info(f"Training {name} classifier")
            model.fit(X_train, y_train)
            
    def predict(self, X, model_name='voting'):
        """Make predictions using specified model."""
        logger.info(f"Making predictions with {model_name} classifier")
        return self.models[model_name].predict(X)
        
    def predict_proba(self, X, model_name='voting'):
        """Get prediction probabilities."""
        return self.models[model_name].predict_proba(X)
        
    def evaluate(self, X_test, y_test, class_names):
        """Evaluate all models and generate reports."""
        logger.info("Evaluating ML classifiers")
        
        results = {}
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Generate classification report
            report = classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True
            )
            
            # Plot confusion matrix
            self._plot_confusion_matrix(
                y_test,
                y_pred,
                class_names,
                title=f"{name.replace('_', ' ').title()} Confusion Matrix",
                save_path=OUTPUTS_DIR / f"{name}_confusion_matrix.png"
            )
            
            results[name] = report
            
        # Plot comparison of model performances
        self._plot_model_comparison(results, class_names)
        
        return results
        
    def _plot_confusion_matrix(self, y_true, y_pred, class_names, title, save_path):
        """Plot confusion matrix for a model."""
        plt.figure(figsize=VIS_CONFIG["plot_size"])
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()
        
    def _plot_model_comparison(self, results, class_names):
        """Plot comparison of model performances."""
        plt.figure(figsize=VIS_CONFIG["plot_size"])
        
        # Extract accuracy scores
        accuracies = {
            name: report['accuracy']
            for name, report in results.items()
        }
        
        # Create bar plot
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=VIS_CONFIG["dpi"])
        plt.close()
        
        # Plot per-class performance
        plt.figure(figsize=(15, 8))
        
        # Extract per-class F1-scores
        f1_scores = {
            name: {
                class_name: report[class_name]['f1-score']
                for class_name in class_names
            }
            for name, report in results.items()
        }
        
        # Create grouped bar plot
        x = np.arange(len(class_names))
        width = 0.2
        multiplier = 0
        
        for model_name, scores in f1_scores.items():
            offset = width * multiplier
            plt.bar(x + offset, scores.values(), width, label=model_name.replace('_', ' ').title())
            multiplier += 1
            
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('Per-class Performance Comparison')
        plt.xticks(x + width, class_names, rotation=45)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        plt.savefig(OUTPUTS_DIR / "per_class_comparison.png", dpi=VIS_CONFIG["dpi"])
        plt.close()
        
    def save_models(self, base_path):
        """Save all trained models."""
        logger.info("Saving ML models")
        
        for name, model in self.models.items():
            save_path = Path(base_path) / f"{name}_model.joblib"
            joblib.dump(model, save_path)
            
    def load_models(self, base_path):
        """Load all trained models."""
        logger.info("Loading ML models")
        
        for name in self.models.keys():
            load_path = Path(base_path) / f"{name}_model.joblib"
            self.models[name] = joblib.load(load_path)
            
        # Update individual model references
        self.rf_clf = self.models['random_forest']
        self.svm_clf = self.models['svm']
        self.lr_clf = self.models['logistic_regression']
        self.voting_clf = self.models['voting'] 