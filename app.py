import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
import easyocr


class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Professional ANPR System (Max Power)")
        self.root.geometry("1280x720")
        self.root.configure(bg="#1e1e1e")

        # --- VARIABLES ---
        self.video_path = "data/traffic.mp4" 
        self.is_running = False
        self.cap = None
        self.frame_count = 0 
        
        # --- 1. LOAD MODELS ---
        print(" Loading Vehicle Model...")
        self.vehicle_model = YOLO('yolov8n.pt') 
        
        print(" Loading Plate Model...")
        self.plate_model_path = "plate_model.pt" 
        self.plate_model = YOLO(self.plate_model_path) if os.path.exists(self.plate_model_path) else None
        
        if self.plate_model:
            print(" License Plate Model Loaded!")
        else:
            print(" WARNING: 'plate_model.pt' not found!")

        print(" Loading OCR...")
        self.reader = easyocr.Reader(['en'], gpu=False) 
        print(" System Ready!")

        # 2. Line Setup
        self.START = sv.Point(0, 350)   
        self.END = sv.Point(1280, 350)
        self.line_zone = sv.LineZone(start=self.START, end=self.END)
        
        # 3. Annotators
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=5)
        self.SELECTED_CLASSES = [2, 3, 5, 7] 

        self.vehicle_plates = {} 

        # 4. CSV Setup
        self.csv_file = "traffic_data_pro.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Vehicle_ID", "Type", "Direction", "License_Plate"])

        # --- GUI LAYOUT ---
        self.create_widgets()

    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg="#252526", height=60)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text=" Professional ANPR Dashboard", 
                 font=("Segoe UI", 18, "bold"), bg="#252526", fg="white").pack(pady=10)

        content_frame = tk.Frame(self.root, bg="#1e1e1e")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.video_label = tk.Label(content_frame, bg="black", text="Ready to Start", fg="gray")
        self.video_label.pack(side="left", fill="both", expand=True, padx=(0, 20))

        stats_frame = tk.Frame(content_frame, bg="#2d2d30", width=350)
        stats_frame.pack(side="right", fill="y")
        
        self.lbl_in = tk.Label(stats_frame, text="IN: 0", font=("Arial", 24, "bold"), 
                               bg="#2d2d30", fg="#4caf50")
        self.lbl_in.pack(pady=20)
        self.lbl_out = tk.Label(stats_frame, text="OUT: 0", font=("Arial", 24, "bold"), 
                                bg="#2d2d30", fg="#f44336")
        self.lbl_out.pack(pady=10)

        tk.Label(stats_frame, text="Live Detections:", font=("Arial", 12, "bold"), 
                 bg="#2d2d30", fg="white").pack(anchor="w", padx=10, pady=(20,5))
        self.log_list = tk.Listbox(stats_frame, bg="#1e1e1e", fg="#00ff00", 
                                    font=("Consolas", 11), height=15, borderwidth=0)
        self.log_list.pack(fill="both", expand=True, padx=10, pady=5)

        btn_frame = tk.Frame(stats_frame, bg="#2d2d30")
        btn_frame.pack(side="bottom", fill="x", pady=20)
        self.btn_start = tk.Button(btn_frame, text="▶ START", font=("Arial", 12, "bold"), 
                                    bg="#007acc", fg="white", command=self.start_thread)
        self.btn_start.pack(fill="x", padx=20, pady=5)
        self.btn_stop = tk.Button(btn_frame, text="⏹ STOP", font=("Arial", 12, "bold"), 
                                   bg="#d32f2f", fg="white", command=self.stop_analysis, 
                                   state="disabled")
        self.btn_stop.pack(fill="x", padx=20, pady=5)

    def log_message(self, message):
        self.log_list.insert(0, message)
        if self.log_list.size() > 50:
            self.log_list.delete(50, tk.END)

    def start_thread(self):
        if not self.is_running:
            self.is_running = True
            self.btn_start.config(state="disabled", bg="#555555")
            self.btn_stop.config(state="normal", bg="#d32f2f")
            self.thread = threading.Thread(target=self.process_video)
            self.thread.daemon = True
            self.thread.start()

    def stop_analysis(self):
        self.is_running = False
        self.btn_start.config(state="normal", bg="#007acc")
        self.btn_stop.config(state="disabled", bg="#555555")

    def read_plate(self, frame, x1, y1, x2, y2):
        """Ultra aggressive OCR with no confidence filters"""
        try:
            h, w, _ = frame.shape
            # Padding (expand region slightly)
            padding = 15 
            y1_crop = max(0, int(y1) - padding)
            y2_crop = min(h, int(y2) + padding)
            x1_crop = max(0, int(x1) - padding)
            x2_crop = min(w, int(x2) + padding)

            car_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            if car_crop.size == 0:
                return None

            plate_crop = None
            global_x1, global_y1, global_x2, global_y2 = 0, 0, 0, 0
            
            # 1. Detect Plate (using plate model)
            if self.plate_model:
                results = self.plate_model(car_crop, verbose=False)
                for r in results:
                    for box in r.boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        
                        # Calculate global coordinates
                        global_x1 = x1_crop + px1
                        global_y1 = y1_crop + py1
                        global_x2 = x1_crop + px2
                        global_y2 = y1_crop + py2
                        
                        # Draw yellow box around detected plate
                        cv2.rectangle(frame, (global_x1, global_y1), (global_x2, global_y2), 
                                      (0, 255, 255), 2)
                        
                        plate_crop = car_crop[py1:py2, px1:px2]
                        break 
            
            # Fallback: if no plate detected, use full car crop (risky but necessary)
            if plate_crop is None:
                plate_crop = car_crop

            # 2. IMAGE PROCESSING (maximize clarity)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Upscale 5x for better OCR
            gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            
            # Contrast boost (enhance black/white distinction)
            gray = cv2.equalizeHist(gray)
            
            # Sharpening (enhance character edges)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)

            # 3. READ (NO CONFIDENCE CHECK - accept anything!)
            result = self.reader.readtext(gray)
            
            best_text = ""
            for (bbox, text, prob) in result:
                # KEY CHANGE: removed confidence threshold
                # Accept any text with more than 2 characters
                text = text.replace(" ", "").replace(".", "").replace("-", "")
                if len(text) > 2:
                    best_text = text
                    break  # Take first result found

            final_text = best_text.upper() if best_text else None

            # Write detected plate text on video frame
            if final_text and self.plate_model and plate_crop is not None:
                cv2.putText(frame, final_text, (global_x1, global_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            return final_text

        except Exception as e:
            print(f"OCR Error: {e}")
            return None

    def process_video(self):
        # Warm-up inference
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.vehicle_model(dummy, verbose=False)
        self.cap = cv2.VideoCapture(self.video_path)

        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_analysis()
                break
            
            self.frame_count += 1
            
            # Process every 3rd frame for performance
            if self.frame_count % 3 != 0:
                continue

            # Vehicle detection and tracking
            results = self.vehicle_model.track(frame, persist=True, conf=0.3, 
                                               iou=0.5, imgsz=640, verbose=False)
            
            detections = sv.Detections.from_ultralytics(results[0])
            detections = (detections[np.isin(detections.class_id, self.SELECTED_CLASSES)] 
                          if len(detections) > 0 else detections)

            if detections.tracker_id is not None:
                crossed_in, crossed_out = self.line_zone.trigger(detections=detections)

                # Process vehicles that crossed the line
                for direction, crossed_list in [("IN", crossed_in), ("OUT", crossed_out)]:
                    if any(crossed_list):
                        for i, crossed in enumerate(crossed_list):
                            if crossed:
                                t_id = detections.tracker_id[i]
                                name = self.vehicle_model.model.names[detections.class_id[i]]
                                box = detections.xyxy[i]
                                
                                # Read plate and draw on frame
                                plate_text = self.read_plate(frame, box[0], box[1], 
                                                             box[2], box[3])
                                final_plate = plate_text if plate_text else "Unknown"
                                self.vehicle_plates[t_id] = final_plate
                                
                                self.log_event(t_id, name, direction, final_plate)

                # Create labels for each tracked vehicle
                labels = []
                for t_id, c_id in zip(detections.tracker_id, detections.class_id):
                    lbl = f"#{t_id}"
                    if t_id in self.vehicle_plates and self.vehicle_plates[t_id] != "Unknown":
                        lbl += f" [{self.vehicle_plates[t_id]}]"
                    labels.append(lbl)

                # Annotate frame
                frame = self.box_annotator.annotate(scene=frame, detections=detections)
                frame = self.label_annotator.annotate(scene=frame, detections=detections, 
                                                      labels=labels)
                self.line_annotator.annotate(frame=frame, line_counter=self.line_zone)

            # Update GUI
            self.update_gui_image(frame)
            self.lbl_in.config(text=f"IN: {self.line_zone.in_count}")
            self.lbl_out.config(text=f"OUT: {self.line_zone.out_count}")

        self.cap.release()

    def log_event(self, t_id, vehicle_name, direction, plate_text):
        """Log vehicle crossing event to CSV and GUI"""
        now_str = datetime.now().strftime("%H:%M:%S")
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([now_str, t_id, vehicle_name, direction, plate_text])
        
        icon = "⬇️" if direction == "IN" else "⬆️"
        color = "#00ff00" if plate_text != "Unknown" else "#aaaaaa"
        self.log_list.insert(0, f"{icon} {vehicle_name} | {plate_text}")
        self.log_list.itemconfig(0, {'fg': color})

    def update_gui_image(self, frame):
        """Update the GUI with processed video frame"""
        frame = cv2.resize(frame, (960, 540))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.mainloop()