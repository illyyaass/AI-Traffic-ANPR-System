import cv2
import csv
import os
import numpy as np # Darori tkon numpy
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

# 1. Chargi l-Model
model = YOLO('yolov8n.pt')

# --- 🔥 FIX: WARMUP (TSKHIN L-MODEL) 🔥 ---
# Hna kan3tiw l-model tswira khawya bach y-demarrer 9bel ma ybda l-video
# Hakka l-video ghadi ybda w l-model deja wajed 100%
print("⏳ Kay-sakhn l-model (Warmup)...")
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
model(dummy_frame, verbose=False)
print("✅ Model Wajed! Bda l-video.")

# 2. Hll l-Video
video_path = "data/traffic.mp4"
cap = cv2.VideoCapture(video_path)

# --- 3. I3dadat dyal l-Khet (Line Setup) ---
START = sv.Point(0, 350)
END = sv.Point(1280, 350)
line_zone = sv.LineZone(start=START, end=END)

# --- Annotators ---
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_padding=10)

# --- 4. CSV Setup ---
csv_file = "traffic_data_detailed.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Vehicle_ID", "Vehicle_Type", "Direction", "Total_IN", "Total_OUT"])

# --- CLASSES ID ---
SELECTED_CLASSES = [2, 3, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- ⚡ FIX: Tracking Parameters ⚡ ---
    # Zidt conf=0.3 (Confidence) bach yz3m 3la detection dghya
    # Zidt iou=0.5 bach yfre9 bin tomobilat mzyan
    results = model.track(frame, persist=True, conf=0.3, iou=0.5)
    
    detections = sv.Detections.from_ultralytics(results[0])

    # Filter Classes
    detections = detections[np.isin(detections.class_id, SELECTED_CLASSES)] if len(detections) > 0 else detections

    if detections.tracker_id is not None:
        
        # Trigger Counting
        crossed_in, crossed_out = line_zone.trigger(detections=detections)
        
        # IN Logic
        if any(crossed_in):
            for i, crossed in enumerate(crossed_in):
                if crossed:
                    class_id = detections.class_id[i]
                    class_name = model.model.names[class_id]
                    tracker_id = detections.tracker_id[i]
                    
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([now, tracker_id, class_name, "IN", line_zone.in_count, line_zone.out_count])
                    print(f"🔻 {class_name} #{tracker_id} Entered! (Total IN: {line_zone.in_count})")

        # OUT Logic
        if any(crossed_out):
            for i, crossed in enumerate(crossed_out):
                if crossed:
                    class_id = detections.class_id[i]
                    class_name = model.model.names[class_id]
                    tracker_id = detections.tracker_id[i]

                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([now, tracker_id, class_name, "OUT", line_zone.in_count, line_zone.out_count])
                    print(f"🔺 {class_name} #{tracker_id} Exited! (Total OUT: {line_zone.out_count})")

        # Rsim
        frame = box_annotator.annotate(scene=frame, detections=detections)
        labels = [f"#{tracker_id} {model.model.names[class_id]}" for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)]
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_zone)

    cv2.imshow("Vehicle Counter System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()