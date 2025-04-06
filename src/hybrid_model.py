from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

def build_hybrid_model():
    # CNN Deep Features Input (2048)
    cnn_input = Input(shape=(2048,), name="deep_features_input")

    # Handcrafted Features Input (8) - Update this if needed
    handcrafted_input = Input(shape=(8,), name="handcrafted_features_input")

    # Concatenating both feature sets
    x = Concatenate(name="concatenated_features")([cnn_input, handcrafted_input])
    
    # Fully connected layers
    x = Dense(256, activation='relu', name="dense_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)  # Regularization
    x = Dense(128, activation='relu', name="dense_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)  # Added dropout to reduce overfitting
    
    # Output layer (adjust classes accordingly)
    output = Dense(10, activation='softmax', name="output_layer")(x)  # Fixed missing connection

    # Create and compile model
    model = Model(inputs=[cnn_input, handcrafted_input], outputs=output, name="Hybrid_Model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
