
# 🥦 Vegetable Image Classifier – Focus on Image Processing

This project focuses on **deep learning-based image processing** to classify vegetables from images with high accuracy. Using **TensorFlow (Keras)** and a **pre-trained InceptionV3 model**, the system performs **real-time vegetable recognition**, while optionally integrating sensor data for inventory management.

---

## 🧠 Project Motivation

Recognizing and classifying vegetables in images is a **challenging computer vision task** due to variations in shape, color, lighting, and background. Accurate image processing enables:

* **Automatic vegetable classification** without human supervision
* **Real-time feedback** for inventory management
* **Reduction of operational errors** caused by manual labeling

By focusing on **image preprocessing, augmentation, and transfer learning**, this project ensures robust classification even with limited datasets.

---

## 📂 Dataset

* **Source:** [Kaggle - Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
* **Classes:** 15 vegetables (e.g., tomato, potato, carrot)
* **Structure:**

```
Vegetable_Images/
├── train/
│   ├── Tomato/
│   ├── Potato/
│   └── ...
└── test/
    ├── Tomato/
    ├── Potato/
    └── ...
```

---

## 🚀 Image Processing Pipeline

1. **Image Loading and Resizing**
   All images are resized to **75x75 pixels** for faster processing while retaining essential features.

2. **Data Augmentation**
   To improve generalization, the system applies:

   * Rotations
   * Zoom
   * Horizontal flips
   * Color normalization

3. **Preprocessing for InceptionV3**
   Images are normalized according to the pre-trained model’s requirements, ensuring **optimal feature extraction**.

4. **Feature Extraction and Classification**

   * **Base model:** InceptionV3 (pre-trained on ImageNet, top layers removed)
   * **Custom layers:** Global Average Pooling → Dense (128, ReLU) → Dropout (0.5) → Dense output (Softmax over 15 classes)

**Transfer learning** allows the system to leverage pre-learned features, focusing on vegetable-specific patterns like texture, shape, and color distribution.

---

## 🏗️ Setup & Usage

1. **Install dependencies**

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

2. **Project structure**

```
vegetable_image_processing/
├── Vegetable_Images/
│   ├── train/
│   └── test/
├── model_training.py
├── image_predictor.py
└── My_test/          # Personal test images
```

3. **Train the model**

```bash
python model_training.py
```

* Loads images and labels
* Performs **data augmentation**
* Trains the model for 10 epochs
* Saves `vegetable_classifier.keras` and `label_map.json`

4. **Predict new images**

```bash
python image_predictor.py
```

* Outputs **top 3 predictions per image** with confidence scores

**Example output:**

```
Image: my_carrot.jpg
Top 3 Predictions:
1. Carrot (0.9453)
2. Tomato (0.0214)
3. Radish (0.0158)
```

---

## 🛠 Technologies Used

* **Python** 🐍
* **TensorFlow / Keras** 🧠 – CNN model for image feature extraction
* **OpenCV** 📸 – image loading, preprocessing, and augmentation
* **NumPy** 🧮 – numerical operations
* **Scikit-learn** 📊 – evaluation metrics

Optional for inventory tracking:

* **ESP32 + HX711** for weight-based triggers
* **React + Node.js** for web interface

---

## 🔍 Key Features

* **Advanced Image Processing:** preprocessing and augmentation for robust recognition
* **High Accuracy Classification:** leverages InceptionV3 pre-trained features
* **Real-time Predictions:** fast inference on new images
* **Transfer Learning:** focuses on vegetable-specific patterns
* **Optional Inventory Integration:** weight sensor triggers camera capture for synchronized stock monitoring

---

## 📌 Notes

* Images can be resized to higher resolutions to improve detection of subtle features
* Augmentation helps the model generalize across different lighting and orientations
* Freezing InceptionV3 layers speeds up training; unfreezing may improve accuracy further

