# 😷 Face Mask Detection using CNN (Convolutional Neural Network)

This repository contains a deep learning project for **binary classification** of face images to detect whether a person is wearing a **face mask or not**. It leverages **Convolutional Neural Networks (CNNs)** using **TensorFlow/Keras** and provides detailed visualizations for both **loss** and **accuracy** across training and validation datasets.

---

## 🚀 Project Objective

Due to the COVID-19 pandemic, face mask detection has become a critical application in public safety systems. This project aims to:

- Build a CNN-based image classifier to detect **face masks in images**.
- Visualize training progress via **loss and accuracy plots**.
- Provide a clean, easy-to-read, and extensible codebase.

---


## 🧠 Model Architecture (CNN)

The model is a **sequential CNN** with the following layers:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers to prevent overfitting
- Flatten and Dense layers for classification
- Final Dense layer with `sigmoid` activation for binary output

> All layers are implemented in `main.py` using Keras.

---

## 📈 Training Results

The model is trained for 5 epochs. The following plots were generated:

### 🔻 Loss Curve
![Loss Curve](https://github.com/nithun-rajan/Face-mask-detection-using-CNN/blob/main/plots/Figure_1.png?raw=true)

### 🔺 Accuracy Curve
![Accuracy Curve](https://github.com/nithun-rajan/Face-mask-detection-using-CNN/blob/main/plots/Figure_2.png?raw=true)


---

## 📊 Observations

- **Train Accuracy**: Improves steadily from ~80% to ~94%.
- **Validation Accuracy**: Starts high (~89%) and stabilizes around ~93%.
- **No significant overfitting observed**: The validation accuracy closely follows the training accuracy.

---

## 🧪 Dataset

The dataset should contain two folders:  
- `with_mask/` – Images of people wearing masks  
- `without_mask/` – Images of people not wearing masks  

If not included, you can create a similar structure using any open dataset (e.g., Kaggle: [Face Mask Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-dataset)).

---

## 📦 Requirements

Install required packages via pip:

```bash
pip install tensorflow numpy matplotlib
Or use a requirements.txt file for easier setup.

💻 How to Run

Clone the repo:
git clone https://github.com/nithun-rajan/Face-mask-detection-using-CNN.git
cd Face-mask-detection-using-CNN
Make sure the dataset is placed under ./data/ in the correct folder structure.
Run the training script:
python main.py
Check the ./plots/ folder for graphs.
🎯 Future Enhancements

✅ Real-time face mask detection using OpenCV
✅ Deploy the model via Streamlit, Flask, or FastAPI
⏳ Use Transfer Learning with MobileNetV2 or ResNet50
⏳ Improve generalization with Data Augmentation
⏳ Export model as .h5 or TensorFlow Lite for edge devices
