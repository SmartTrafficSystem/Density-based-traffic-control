import time
import joblib
import numpy as np
import pandas as pd
from yolo_plus_cv import lane_vehicle_data  # ✅ IMPORT RESULT FROM PHASE 5

# -------------------------------
# LOAD TRAINED ML MODEL
# -------------------------------
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# -------------------------------
# SIGNAL STATES
# -------------------------------
signals = {
    "Lane_1": "RED",
    "Lane_2": "RED",
    "Lane_3": "RED"
}

LANES = list(signals.keys())  # ✅ ROUND ROBIN ORDER
current_lane_index = 0        # ✅ ROUND ROBIN POINTER

# -------------------------------
# FEATURE PREPARATION
# -------------------------------
def prepare_features(vehicle_data):
    return pd.DataFrame([{
        "car": vehicle_data["car"],
        "bike": vehicle_data["bike"],
        "bus": vehicle_data["bus"],
        "truck": vehicle_data["truck"]
    }])

# -------------------------------
# ML GREEN TIME PREDICTION
# -------------------------------
def predict_green_time(features):
    features = pd.DataFrame([features])
    features = features.rename(columns={
        'car': 'cars',
        'bus': 'buses',
        'bike': 'bikes',
        'truck': 'trucks'
    })

    if 'total' not in features.columns:
        features['total'] = features[['cars','buses','bikes','trucks']].sum(axis=1)

    transformed_features = pipeline.transform(features)
    green_time = round(model.predict(transformed_features)[0])

    return max(10, min(green_time, 120))

# -------------------------------
# SIGNAL CONTROLLER (ROUND ROBIN)
# -------------------------------
def run_signal_cycle():
    global lane_vehicle_data, current_lane_index

    if not lane_vehicle_data:
        print("⚠️ No data received from Phase 5")
        return

    # ✅ ROUND ROBIN LANE SELECTION
    current_lane = LANES[current_lane_index]
    current_lane_index = (current_lane_index + 1) % len(LANES)

    green_time = predict_green_time(lane_vehicle_data[current_lane])

    # Reset signals
    for lane in signals:
        signals[lane] = "RED"

    signals[current_lane] = "GREEN"

    print("\n==============================")
    print("🚦 ADAPTIVE TRAFFIC SIGNAL (ROUND ROBIN)")
    print("==============================")

    for lane, data in lane_vehicle_data.items():
        print(f"{lane}: {data}")

    print(f"\n🟢 {current_lane} GREEN for {green_time} seconds\n")

    while green_time > 0:
        print(f"{current_lane} | GREEN | {green_time}s")
        time.sleep(1)
        green_time -= 1

    print(f"{current_lane} | YELLOW")
    time.sleep(3)

    signals[current_lane] = "RED"
    print(f"{current_lane} | RED")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    run_signal_cycle()
    print("\n🔄 Waiting for next YOLO update...\n")
    time.sleep(5)
