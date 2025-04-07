"""
ResNet model module for plant disease classification.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import logging
from pathlib import Path
from .config import MODEL_CONFIG, OUTPUTS_DIR, LOGGING_CONFIG

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ResNetModel:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        """Initialize ResNet model."""
        logger.info("Initializing ResNet model")
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
    def compile(self, learning_rate=0.001):
        """Compile the model."""
        logger.info("Compiling ResNet model")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, train_generator, validation_generator, epochs=10, callbacks=None):
        """Train the model."""
        logger.info("Training ResNet model")
        
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
        
    def evaluate(self, test_generator):
        """Evaluate the model."""
        logger.info("Evaluating ResNet model")
        return self.model.evaluate(test_generator)
        
    def predict(self, X):
        """Make predictions."""
        logger.info("Making predictions with ResNet")
        return self.model.predict(X)
        
    def save_model(self, path):
        """Save the model."""
        logger.info(f"Saving model to {path}")
        self.model.save(path)
        
    def load_model(self, path):
        """Load a saved model."""
        logger.info(f"Loading model from {path}")
        self.model = tf.keras.models.load_model(path)
        
    def unfreeze_layers(self, num_layers=20):
        """Unfreeze some layers for fine-tuning."""
        logger.info(f"Unfreezing last {num_layers} layers")
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True
            
    def get_model_summary(self):
        """Get model summary."""
        return self.model.summary()
        
    def plot_model(self, path=None):
        """Plot model architecture."""
        logger.info("Plotting model architecture")
        if path is None:
            path = OUTPUTS_DIR / "resnet_architecture.png"
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(path),
            show_shapes=True,
            show_layer_names=True
        ) 