from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Build a GFNet model
def build_model(input_shape=(224,224,3)):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg')
    x = base_model.output
    # Gradient usually use "relu"
    x = layers.Dense(128, activation='relu')(x) #128 output dimensions (neurons)
    x = layers.Dropout(0.5)(x)
    # To solve a binary classification problem
    # softmax/sigmod - focus on binary or multi-classification/just 2-classification
    predictions = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model