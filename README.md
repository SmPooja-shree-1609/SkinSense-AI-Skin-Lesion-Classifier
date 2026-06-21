# SkinSense: AI-Based Skin Lesion Classifier

A Streamlit-based AI application that uses deep learning to classify skin lesion images using a trained EfficientNet model.

---

## Overview

SkinSense is an AI-powered skin lesion classification system designed to analyze skin images and predict the corresponding lesion category.

The application uses a deep learning model to perform image classification and provides the predicted class along with the confidence score through an interactive web interface.

---

## Features

- 🩺 AI-based skin lesion classification
- 🧠 Deep learning model powered by EfficientNet
- 📷 Image upload and real-time prediction
- 📊 Prediction confidence score
- ⚡ Fast inference using TensorFlow/Keras
- 🌐 User-friendly Streamlit interface

---

## Project Structure

```text
SkinSense/
│
├── app.py
├── predictor.py
├── verify_env.py
├── requirements.txt
├── README.md
│
└── models/
    └── best_modelnew.h5
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNet
- Streamlit
- NumPy
- Pillow
- Scikit-learn

---

## Installation

### Clone the repository

```bash
git clone https://github.com/SmPooja-shree-1609/SkinSense-AI-Skin-Lesion-Classifier.git
```

### Navigate to project folder

```bash
cd SkinSense-AI-Skin-Lesion-Classifier
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Setup

The trained model file is not included in this repository because of its size.

Download the model:

Google Drive:

https://drive.google.com/file/d/1G8URGaJZmo2mn3XdZ0QsOpdQXWwXZa_M/view?usp=sharing

After downloading, place it inside:

```text
models/best_modelnew.h5
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## How Prediction Works

1. User uploads a skin lesion image.
2. Image preprocessing is performed.
3. Image is resized to 224 × 224 pixels.
4. Pixel values are normalized.
5. EfficientNet model performs classification.
6. The predicted class with highest probability is returned.
7. Confidence score is displayed.

---

## Model Workflow

```text
Input Image
      |
      ↓
Image Preprocessing
      |
      ↓
EfficientNet Deep Learning Model
      |
      ↓
Class Prediction
      |
      ↓
Confidence Score
```

---

## Model Performance

The model was evaluated on a test dataset containing **1502 images**.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| akiec | 1.00 | 0.22 | 0.37 | 49 |
| bcc | 0.68 | 0.58 | 0.63 | 77 |
| bkl | 0.51 | 0.77 | 0.62 | 165 |
| df | 0.50 | 0.24 | 0.32 | 17 |
| mel | 0.56 | 0.44 | 0.49 | 167 |
| nv | 0.90 | 0.91 | 0.91 | 1006 |
| vasc | 0.73 | 0.76 | 0.74 | 21 |

---

## Overall Metrics

```
Accuracy  : 0.79

Macro Average:
Precision : 0.70
Recall    : 0.56
F1-score  : 0.58

Weighted Average:
Precision : 0.81
Recall    : 0.79
F1-score  : 0.79
```

---

## Important Notes

- Ensure the trained model file is placed correctly before running.
- Use compatible TensorFlow and Keras versions.
- This project is intended for educational and research purposes.

---

## Future Improvements

- Improve accuracy using larger balanced datasets.
- Add Explainable AI techniques like Grad-CAM.
- Deploy using cloud platforms.
- Improve performance on minority classes.

---

## Disclaimer

This project is developed for academic and learning purposes only.

It is not a replacement for professional medical diagnosis or clinical decision-making.

---

## Author

**Pooja Shree**
