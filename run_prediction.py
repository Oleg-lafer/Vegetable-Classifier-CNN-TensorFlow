import cv2
import numpy as np
import json
import os
from tensorflow import keras

# Load the model
loaded_model = keras.models.load_model("vegetable_model.keras")

# Print model summary to understand input requirements
loaded_model.summary()

# Load the label mapping
with open("class_indices.json", "r") as f:
    label_map = json.load(f)


def predict_image(image_path, label_map):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    # Convert BGR to RGB (OpenCV reads in BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 150x150 pixels
    img = cv2.resize(img, (150, 150))

    # Normalize pixel values
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Perform prediction
    prediction = loaded_model.predict(img)[0]

    # Sort results from highest to lowest match
    top_indices = np.argsort(prediction)[::-1][:3]

    # Print the primary identification
    predicted_label = list(label_map.keys())[top_indices[0]]
    print(f"Image: {os.path.basename(image_path)} -> Predicted label: {predicted_label}")

    # Print top 3 matches with confidence scores
    print("Top 3 matches:")
    for i in top_indices:
        vegetable_name = list(label_map.keys())[i]
        confidence_score = prediction[i]
        print(f"{vegetable_name}: {confidence_score:.4f}")
    print("-" * 40)


# Directory path for testing
directory = r"My_test"

# Iterate through files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        predict_image(file_path, label_map)
