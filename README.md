# 🚗 Overtaking Risk Prediction System

## 📌 Overview

This project predicts whether it is safe or unsafe to overtake a vehicle using computer vision and deep learning.

The system processes video input, detects surrounding vehicles, estimates motion dynamics, and evaluates overtaking risk using a CNN-LSTM model.

---

## ⚙️ Features

* 🚘 Vehicle detection using YOLOv8
* 📏 Distance estimation
* 🚀 Speed estimation of oncoming vehicles
* 🧠 Risk prediction using CNN-LSTM
* 🎥 Real-time video processing with visual overlays

---

## 🧠 Model Architecture

* **CNN (ResNet18)** → extracts spatial features from frames
* **LSTM** → captures temporal relationships across frame sequences
* Outputs a **risk score (safe / unsafe overtaking)**

---

## 🐍 Requirements

* Python **3.10 recommended**

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Add input video

Place your video inside:

```
input_videos/
```

### 2. Run the project

```bash
python main.py
```

---

## 📦 Model Weights

The trained model is automatically downloaded when you run the project.

If the download fails, you can manually download it here:
https://drive.google.com/file/d/1NeIMkbwoZ-wWIpK7PXM8UPpEX_a_ryG0/view

Place it inside:

```
models/cnn_lstm_risk_40epoch.pth
```

---

## 📁 Project Structure

```
overtaking-risk-prediction/
│
├── main.py
├── models/
│   └── sequence_model.py
├── utils/
│   ├── distance_estimation.py
│   └── speed_estimation.py
├── input_videos/
├── output_videos/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Notes

* Lane and curve detection were experimented with but excluded from the final system due to inconsistent performance.
* The model was trained on sequential video frames to capture temporal dynamics of overtaking scenarios.

---

