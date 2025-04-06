import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ Error enabling GPU memory growth: {e}")

# Load data
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

# Shuffle data before splitting
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# Normalize images (scale to [0,1])
X = X / 255.0

# Hyperparameters
num_classes = len(np.unique(y))
epochs = 10  # For InceptionV3
resnet_epochs = 5  # Fewer epochs for ResNet
batch_size = 32
learning_rate = 0.0001

# Data Augmentation
datagen = ImageDataGenerator(validation_split=0.2)
train_gen = datagen.flow(X, y, batch_size=batch_size, shuffle=True, subset='training')
val_gen = datagen.flow(X, y, batch_size=batch_size, shuffle=True, subset='validation')

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

def build_cnn_model(base_model_name='InceptionV3'):
    if base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    base_model.trainable = False  # Freeze base model weights

    # Custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train InceptionV3 if not already trained
inception_model_path = "models/inception_v3_best.h5"
if os.path.exists(inception_model_path):
    print("✅ Inception-V3 model already trained. Skipping training.")
    inception_model = load_model(inception_model_path)
else:
    print("Training Inception-V3...")
    inception_model = build_cnn_model('InceptionV3')
    inception_checkpoint = ModelCheckpoint(inception_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    inception_model.fit(
        train_gen, epochs=epochs, validation_data=val_gen,
        callbacks=[early_stopping, reduce_lr, inception_checkpoint]
    )
    print("✅ Inception-V3 model training complete!")

# Train ResNet (always trains after InceptionV3)
print("Training ResNet...")
resnet_model = build_cnn_model('ResNet50')
resnet_checkpoint = ModelCheckpoint("models/resnet_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
resnet_model.fit(
    train_gen, epochs=resnet_epochs, validation_data=val_gen,
    callbacks=[early_stopping, reduce_lr, resnet_checkpoint]
)
print("✅ ResNet model training complete!")
