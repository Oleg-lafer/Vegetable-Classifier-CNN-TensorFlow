import cv2
import os
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Load images and resize them to 75x75, which is suitable for InceptionV3
def load_images_from_folder(folder):
    images, labels = [], []
    label_map = {}
    categories = os.listdir(folder)
    for idx, label in enumerate(categories):
        label_map[label] = idx

    for label, idx in label_map.items():
        subfolder_path = os.path.join(folder, label)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (75, 75))  # Resize to 75x75
                    images.append(img)
                    labels.append(idx)

    images_normalized = np.array(images) / 255.0
    labels_array = np.array(labels)
    return images_normalized, labels_array, label_map

X_train, y_train, label_map = load_images_from_folder('Vegetable_Images/train')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_test, y_test, _ = load_images_from_folder('Vegetable_Images/test')

num_classes = len(label_map)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load InceptionV3 without the top layers (input size set to 75x75)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
base_model.trainable = False  # Freeze the base model weights

# Add custom layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)  # Dropout layer to prevent overfitting
output = layers.Dense(num_classes, activation="softmax")(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=output)

# Compile and train the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save("vegetable_classifier.keras")

# Load the trained model for prediction
loaded_model = keras.models.load_model("vegetable_classifier.keras")

# Save the label map
with open("../cam_veg/label_map.json", "w") as f:
    json.dump(label_map, f)
