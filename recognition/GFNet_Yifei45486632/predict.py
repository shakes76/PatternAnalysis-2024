# Import
import tensorflow as tf
from dataset import load_images
from sklearn.preprocessing import LabelEncoder

# Loading
model = tf.keras.models.load_mode('best_model')
test_images, test_labels = load_images('test')

# trans the label of test set to int
encoder = LabelEncoder()
test_labels = encoder.fit_transform(test_labels)
test_labels = tf.keras.utils.to_gategorical(test_labels, 2)

# Predict
predictions = model.predict(test_images)
predicted_classes = predictions.argmax(axis=1)

# Compare: predict result / real result
for i, predicted_class in enumerate(predicted_classes[:10]):
    print(f"Image {i + 1}: Predicted Class = {predicted_class}, Actual Class = {test_labels[i].argmax()}")