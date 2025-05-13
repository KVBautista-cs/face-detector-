# 👁️ Real-Time Face Detector using ResNet50 (MATLAB)

This project uses **transfer learning with ResNet50** to detect and classify faces from a webcam feed in real time using **MATLAB**.

---

## 🚀 Features
- 📷 **Live Webcam Classification** using `vision.CascadeObjectDetector`
- 🧠 **Transfer Learning** on ResNet50 with a custom image dataset
- 🎯 Classifies detected faces into custom categories based on training data
- 📊 Includes live bounding box and label display using `insertObjectAnnotation`

---

## 🛠️ Technologies Used
- MATLAB (Deep Learning Toolbox)
- ResNet50 (pretrained model)
- ImageDatastore + AugmentedImageDatastore
- Webcam integration
- Cascade Object Detector

---

## 🧪 How It Works

1. **Prepare Dataset**
   - Images are stored in labeled folders
   - Loaded using `imageDatastore()` and split into training/validation

2. **Retrain ResNet50**
   - Last layers replaced with custom fully connected + classification layers
   - Trained using `trainNetwork()` with `trainingOptions()`

3. **Run Detection**
   - Starts webcam feed
   - Detects face with `vision.CascadeObjectDetector`
   - Crops and resizes face, classifies it, and displays the result live

---

## 📸 Live Demo Output

> Once started, the webcam shows a live feed and classifies each detected face like this:

