# 🚗 AUTONOMOUS DRIVING ASSISTANCE SYSTEM USING YOLOV8

## 📌 FEATURES

- REAL-TIME OBJECT DETECTION USING YOLOV8
- DYNAMIC ACTION DECISION LOGIC BASED ON OBJECT POSITION
- HIGH-CONFIDENCE DETECTION FOR IMPROVED ACCURACY
- REAL-TIME WEBCAM INPUT AND VISUAL FEEDBACK


## 🧠 How It Works
Object Detection:
The system uses the yolov8m.pt model to detect objects like cars, people, bicycles, trucks, and traffic signs in real-time.

Decision-Making Logic:

Assigns priority to objects based on type and proximity.

Uses object position (left, center, right) to decide:

Turn Left, Turn Right, or Brake/Reverse.

Follows Traffic Rules for signs and lights.

## Real-Time Visualization:

Bounding boxes are color-coded:

🔴 Red = High danger (pedestrians, bikes)

🟠 Orange = Medium danger (vehicles)

🟢 Green = Safe

Displays current decision on screen.


## 🛠️ TECH STACK

- PYTHON
- YOLOV8 (ULTRALYTICS)
- PYTORCH
- OPENCV
- NUMPY

## 🚀 GETTING STARTED

### PREREQUISITES

pip install torch torchvision torchaudio
pip install opencv-python
pip install ultralytics

## RUN THE PROJECT

python car.py

## 📁 PROJECT STRUCTURE

├── car.py
├── README.md

