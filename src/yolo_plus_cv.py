# =========================
# FIX OPENMP ISSUE
# =========================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# IMPORTS
# =========================
import cv2
import numpy as np
import time
from ultralytics import YOLO
import cvzone
import joblib
import pandas as pd

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("yolov8n.pt")
names = model.names

# =========================
# VIDEO PATHS & LANES
# =========================
videos = [
    r"C:\Users\Akanksha\Desktop\adaptiv_traffic_timer\videos\cam_01.mp4",
    r"C:\Users\Akanksha\Desktop\adaptiv_traffic_timer\videos\cam_02.mp4",
    r"C:\Users\Akanksha\Desktop\adaptiv_traffic_timer\videos\cam_03.mp4"
]

lane_1 = np.array([(30, 359), (300, 64), (370, 60), (530, 319)], np.int32)
lane_2 = np.array([(150, 219), (370, 70), (450, 75), (600, 193)], np.int32)
lane_3 = np.array([(195, 300), (250, 80), (500, 107), (700, 300)], np.int32)
lanes = [lane_1, lane_2, lane_3]

LANES = ["Lane_1","Lane_2","Lane_3"]
signals = {lane:"RED" for lane in LANES}
lane_vehicle_data = {lane:{"car":0,"bike":0,"bus":0,"truck":0,"total":0} for lane in LANES}
current_lane_index = 0

# =========================
# LOAD ML MODEL
# =========================
ml_model = joblib.load("model.pkl")
ml_pipeline = joblib.load("pipeline.pkl")

# =========================
# UTILITIES
# =========================
def inside_lane(point, lane):
    x, y = int(point[0]), int(point[1])
    lane = lane.reshape((-1,1,2))
    return cv2.pointPolygonTest(lane, (x, y), False) >= 0

def predict_green_time(vehicle_data):
    features = pd.DataFrame([vehicle_data]).rename(columns={
        "car":"cars","bike":"bikes","bus":"buses","truck":"trucks"
    })
    features["total"] = features.sum(axis=1)
    transformed = ml_pipeline.transform(features)
    green_time = round(ml_model.predict(transformed)[0])
    return max(10, min(green_time, 120))

# =========================
# PROCESS VIDEO FOR ONE LANE
# =========================
def process_video(cam_index):
    cap = cv2.VideoCapture(videos[cam_index])
    lane = lanes[cam_index]
    counted_ids = set()
    vehicle_counts = {"car":0,"bike":0,"bus":0,"truck":0,"total":0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(1020,600))
        results = model.track(frame, persist=True, classes=[2,3,5,7], conf=0.4)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                x1,y1,x2,y2 = box
                label = names[cls_id]
                if label=="motorcycle": label="bike"
                cx,cy = (x1+x2)//2,(y1+y2)//2

                if inside_lane((cx,cy), lane) and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_counts[label]+=1
                    vehicle_counts["total"]+=1

                # Draw detection
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cvzone.putTextRect(frame,f"{label}-{track_id}",(x1,y1-5),1,1)

        cv2.polylines(frame,[lane],True,(255,0,0),3)

        # Display counts
        y=40
        for k,v in vehicle_counts.items():
            cvzone.putTextRect(frame,f"{k.upper()}:{v}",(20,y),2,2)
            y+=40

        lane_vehicle_data[LANES[cam_index]] = vehicle_counts

        cv2.imshow(f"Cam {cam_index+1}", frame)
        if cv2.waitKey(1) & 0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return vehicle_counts

# =========================
# ROUND ROBIN SIGNAL LOOP
# =========================
print("🚦 Adaptive Traffic Signal with Sequential Video Processing Started")

while True:
    # -------------------------
    # Current lane
    current_lane = LANES[current_lane_index]
    print(f"\nProcessing {current_lane} video and predicting green time...")

    # Process current lane video
    vehicle_counts = process_video(current_lane_index)

    # Predict green time for this lane
    green_time = predict_green_time(vehicle_counts)

    # Set signals
    for lane in signals: signals[lane] = "RED"
    signals[current_lane] = "GREEN"
    print(f"\n🟢 {current_lane} GREEN for {green_time}s | Vehicles: {vehicle_counts}")

    # Predict next lane's green time while current lane is counting down
    next_index = (current_lane_index + 1) % len(LANES)
    next_lane = LANES[next_index]
    if lane_vehicle_data[next_lane]:
        next_green_time = predict_green_time(lane_vehicle_data[next_lane])

    # Countdown for current green
    for t in range(green_time,0,-1):
        print(f"{current_lane} | GREEN | {t}s")
        time.sleep(1)

    print(f"{current_lane} | YELLOW")
    time.sleep(3)
    signals[current_lane] = "RED"
    print(f"{current_lane} | RED")

    # Move to next lane
    current_lane_index = (current_lane_index + 1) % len(LANES)
