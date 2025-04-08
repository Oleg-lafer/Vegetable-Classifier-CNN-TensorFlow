
```markdown
# 🥦 Vegetable Image Classifier with InceptionV3

This project is a deep learning-based image classification system for recognizing vegetables from images using **TensorFlow (Keras)** and a **pre-trained InceptionV3 model**. It was trained on the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) from Kaggle.

---

## 📂 Dataset

- **Source:** [Kaggle - Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Description:** The dataset contains images of 15 vegetable classes such as tomato, potato, carrot, etc.
- **Structure:**
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

## 🧠 Model Architecture

- **Base model:** InceptionV3 (pre-trained on ImageNet, top layers removed)
- **Custom layers:**
  - Global Average Pooling
  - Dense (128 neurons, ReLU)
  - Dropout (0.5)
  - Dense output layer (Softmax over 15 classes)
- **Image size:** All images are resized to **75x75** pixels to reduce computation time.

---

## 🚀 How to Use

### 1. 🏗️ Setup

Make sure you have Python 3.x installed, then install the required packages:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### 2. 📁 Directory Structure

Place your project files like this:

```
vegetable_image_processing/
├── Vegetable_Images/
│   ├── train/
│   └── test/
├── model_training.py
├── image_predictor.py
└── My_test/          # Your personal test images
```

> ✅ **Note:** Change `"My_test"` folder path if you want to test your own images.

### 3. 🏋️‍♂️ Train the Model

```bash
python model_training.py
```

This will:
- Load the training images and labels
- Train the model for 10 epochs
- Save the trained model as `vegetable_classifier.keras`
- Save the label mapping to `label_map.json`

### 4. 🔍 Predict New Images

Put your images in the `My_test` folder and run:

```bash
python image_predictor.py
```

This script will:
- Load your saved model and label map
- Predict the top 3 vegetable classes for each image
- Print prediction results with confidence scores

---

## 🧪 Example Output

```
📷 Image: my_carrot.jpg
🔝 Top 3 Predictions:
1. Carrot (0.9453)
2. Tomato (0.0214)
3. Radish (0.0158)
----------------------------------------
```

---

## 🛠 Technologies Used

- Python 🐍
- TensorFlow / Keras 🧠
- OpenCV 📸
- NumPy 🧮
- Scikit-learn 📊

---

## 📌 Notes

- InceptionV3 was used with frozen weights for fast and effective transfer learning.
- You can increase the image resolution or unfreeze layers for better accuracy (at the cost of speed).

---

## 📬 Credits

- Dataset by [Ahmed Misrak](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- Project by [Oleg Muraviov]

---

## ⭐️ Give it a star if you find it helpful!
```

---

אם אתה רוצה שאכין גם גרסה בעברית או קובץ PDF/HTML אסתטי מה-README – תגיד ונסדר את זה 💡
