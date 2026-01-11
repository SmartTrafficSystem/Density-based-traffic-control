# Adaptive Traffic Signal Control System using YOLO and Machine Learning

## 📌 Project Overview

This project implements an **Adaptive Traffic Signal Control System** that dynamically adjusts traffic light green time based on **real-time vehicle density**. It combines **computer vision (YOLOv8 + OpenCV)** for vehicle detection and **machine learning** for intelligent green-time prediction.

The system aims to reduce congestion, improve traffic flow efficiency, and outperform traditional fixed-time or round-robin traffic signal methods.

---

## 🎯 Objectives

* Detect and classify vehicles (cars, buses, trucks, bikes) from traffic videos
* Assign higher priority to heavy vehicles (buses & trucks)
* Dynamically predict optimal green signal duration
* Compare adaptive approach with traditional round-robin logic

---

## 🧠 Technologies Used

* **Python 3**
* **YOLOv8 (Ultralytics)** – Vehicle detection
* **OpenCV** – Video processing
* **Scikit-learn** – Machine Learning model
* **Pandas & NumPy** – Data handling
* **Joblib** – Model persistence

---

## 📂 Project Structure

```
Adaptive_Traffic_Signal_System/
│
├── videos/                         # Input traffic videos
│
├── dataset_generation_code.py      # Dataset generation with weighted vehicles
├── ml_training.py                  # ML model training
├── yolo_plus_cv.py                 # YOLO + OpenCV vehicle detection
├── round_robin.py                  # Traditional traffic signal logic
│
├── input.csv                       # Raw vehicle data
├── output.csv                      # Processed data
├── traffic_green_time_dataset.csv  # Final ML dataset
│
├── model.pkl                       # Trained ML model
├── pipeline.pkl                    # Preprocessing + ML pipeline
├── yolov8n.pt                      # YOLO model weights
│
├── main_old.py                     # Older implementation (archived)
└── README.md
```

---

## ⚙️ How It Works

1. **Vehicle Detection**
   YOLOv8 detects vehicles from traffic video feeds and classifies them into categories.

2. **Weighted Vehicle Count**

   * Truck & Bus → High weight
   * Car → Medium weight
   * Bike → Low weight

3. **Dataset Generation**
   Vehicle counts are converted into a dataset used for training the ML model.

4. **ML-Based Green Time Prediction**
   The trained model predicts the optimal green signal duration based on traffic density.

5. **Comparison with Round Robin**
   Adaptive results are compared against fixed-time round-robin logic.

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install ultralytics opencv-python scikit-learn pandas numpy joblib cvzone
```

### 2️⃣ Run Vehicle Detection

```bash
python yolo_plus_cv.py
```

### 3️⃣ Generate Dataset

```bash
python dataset_generation_code.py
```

### 4️⃣ Train ML Model

```bash
python ml_training.py
```

---

## 📊 Results

* Improved traffic flow during peak hours
* Reduced waiting time for heavy vehicles
* More efficient green-time allocation than fixed timers

---

## 🔮 Future Enhancements

* Multi-lane and multi-junction support
* Emergency vehicle detection
* Integration with IoT-based traffic lights
* Real-time deployment using edge devices

---

## 👩‍💻 Author

**Akanksha Agre**
Final Year Engineering Student

**Shruti Adsul**
Final Year Engineering Student


---

## 📜 License

This project is for academic and research purposes.
