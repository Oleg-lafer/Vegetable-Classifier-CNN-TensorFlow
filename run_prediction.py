import cv2
import numpy as np
import json
import os
from tensorflow import keras

# Load the trained model
loaded_model = keras.models.load_model("vegetable_classifier.keras")

# Load the label map
with open("../cam_veg/label_map.json", "r") as f:
    label_map = json.load(f)

# Invert the label map so that indices are keys and names are values
index_to_label = {v: k for k, v in label_map.items()}


def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    img = cv2.resize(img, (75, 75)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Perform prediction
    prediction = loaded_model.predict(img)[0]  # Take the first array since it's a single image

    # Sort predictions by confidence score in descending order
    top_indices = np.argsort(prediction)[::-1][:3]  # Take the top 3 predictions

    print(f"\n📷 Image: {os.path.basename(image_path)}")
    print("🔝 Top 3 Predictions:")

    for i, idx in enumerate(top_indices):
        vegetable_name = index_to_label.get(idx, "Unknown")
        confidence_score = prediction[idx]
        print(f"{i + 1}. {vegetable_name} ({confidence_score:.4f})")

    print("-" * 40)


# Directory path for testing
directory = "C:\\Users\\olegl\\PycharmProjects\\vegetables_image_processing\\My_test"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        predict_image(file_path)
