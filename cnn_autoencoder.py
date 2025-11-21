import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import json
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Paths to datasets
train_zip_path = '/content/drive/My Drive/train_actor_faces.zip'
validate_zip_path = '/content/drive/My Drive/validate_actor_faces.zip'
test_zip_path = '/content/drive/My Drive/test_actor_faces.zip'

# Unzipping datasets
for zip_path, output_dir in [(train_zip_path, 'train_dataset'),
                             (validate_zip_path, 'validate_dataset'),
                             (test_zip_path, 'test_dataset')]:

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

# Directory paths
train_dir = 'train_dataset/train_actor_faces'
validate_dir = 'validate_dataset/validate_actor_faces'
test_dir = 'test_dataset/test_actor_faces'

# Data preprocessing and augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
validate_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Update target_size to (32, 32)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),  
    batch_size=32,
    class_mode='categorical')

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(32, 32),  
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),  # Changed from (32, 32) to (50, 50)
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

class LCNLayers(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, epsilon=1e-5, **kwargs):
        super(LCNLayers, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def call(self, inputs):
        mean = tf.nn.avg_pool2d(inputs, ksize=self.kernel_size, strides=1, padding='SAME')
        squared_mean = tf.nn.avg_pool2d(inputs**2, ksize=self.kernel_size, strides=1, padding='SAME')
        stddev = tf.sqrt(squared_mean - mean**2 + self.epsilon)
        return (inputs - mean) / (stddev + self.epsilon)

    def get_config(self):
        config = super(LCNLayers, self).get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "epsilon": self.epsilon,
        })
        return config


def build_cnn_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)

    # First Convolutional Layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = LCNLayers()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Second Convolutional Layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = LCNLayers()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Third Convolutional Layer
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = LCNLayers()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Latent Space
    encoded = Dropout(0.3)(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = LCNLayers()(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = LCNLayers()(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = LCNLayers()(x)
    x = UpSampling2D((2, 2))(x)

    # Output Layer
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Models
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    return autoencoder, encoder




input_shape = (32, 32, 3)  
cnn_autoencoder, cnn_encoder = build_cnn_autoencoder(input_shape)
cnn_autoencoder.compile(optimizer='adam', loss='mse')

# Prepare data for autoencoder
X_train = np.concatenate([train_generator[i][0] for i in range(len(train_generator))])
y_train = np.concatenate([train_generator[i][1] for i in range(len(train_generator))])

X_val = np.concatenate([validate_generator[i][0] for i in range(len(validate_generator))])
y_val = np.concatenate([validate_generator[i][1] for i in range(len(validate_generator))])

X_test = np.concatenate([test_generator[i][0] for i in range(len(test_generator))])
y_test = np.concatenate([test_generator[i][1] for i in range(len(test_generator))])

# Train Autoencoder
history = cnn_autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=20,
    batch_size=32,
    shuffle=True
)

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Use Encoder for Feature Extraction
train_projected = cnn_encoder.predict(X_train)
val_projected = cnn_encoder.predict(X_val)
test_projected = cnn_encoder.predict(X_test)

# Flatten the feature maps
train_projected = train_projected.reshape(train_projected.shape[0], -1)
val_projected = val_projected.reshape(val_projected.shape[0], -1)
test_projected = test_projected.reshape(test_projected.shape[0], -1)

# Define the SVM parameter grid for GridSearchCV
param_grid = {
    'C': [10],
    'gamma': ['auto'],
    'kernel': ['rbf']
}

# Train SVM using GridSearchCV
print("Performing GridSearchCV to find the best SVM parameters...")
grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2)
grid_search.fit(train_projected, np.argmax(y_train, axis=1))

# Provide feedback during training
print("SVM training completed.")
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Test the model on the validation and test data
y_val_pred = grid_search.best_estimator_.predict(val_projected)
y_test_pred = grid_search.best_estimator_.predict(test_projected)

val_accuracy = accuracy_score(np.argmax(y_val, axis=1), y_val_pred)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)

print(f"Validation accuracy of the best SVM model: {val_accuracy * 100:.2f}%")
print(f"Test accuracy of the best SVM model: {test_accuracy * 100:.2f}%")

# Visualize SVM Classification Results
print("Visualizing SVM Classification Results on Test Data:")
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(np.clip(X_test[i], 0, 1))
    ax.set_title(f"Pred: {y_test_pred[i]}, True: {np.argmax(y_test[i])}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Save the Autoencoder Model
autoencoder_path = 'cnn_autoencoder_model.h5'
cnn_autoencoder.save(autoencoder_path)
print(f"Autoencoder model saved to {autoencoder_path}")

# Save the SVM Model
svm_model_path = 'svm_model.pkl'
joblib.dump(grid_search.best_estimator_, svm_model_path)
print(f"SVM model saved to {svm_model_path}")

# Final Summary
print("\nProject Summary:")
print(f"Classification Accuracy:")
print(f"  Validation: {val_accuracy * 100:.2f}%")
print(f"  Test: {test_accuracy * 100:.2f}%")

# Optional: Save Results and Metrics
results = {
    'val_accuracy': val_accuracy * 100,
    'test_accuracy': test_accuracy * 100,
    'best_svm_params': grid_search.best_params_,
}

results_path = 'results.json'
with open(results_path, 'w') as f:
    json.dump(results, f)
print(f"Results saved to {results_path}")
