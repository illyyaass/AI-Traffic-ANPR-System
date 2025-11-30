#  AI Traffic & ANPR Monitoring System

An advanced Computer Vision project that performs real-time **Vehicle Counting** and **Automatic Number Plate Recognition (ANPR)**. Built with Python, YOLOv8, and EasyOCR, featuring a modern Dark Mode Dashboard.


##  Key Features

- ** Real-Time Vehicle Detection:** Accurately detects Cars, Trucks, Buses, and Motorcycles using `YOLOv8`.
- ** Aggressive License Plate Recognition:** Uses `EasyOCR` with advanced image preprocessing (thresholding, zooming) to force-read plates even in low resolution.
- ** Traffic Counting:** Automated logic to count vehicles exiting the frame (OUT Counting).
- ** Professional Dashboard:** A custom GUI built with `Tkinter`, featuring:
  - Live Video Feed with Bounding Boxes.
  - Real-time "OUT" Counter.
  - Live Activity Log (ID | Type | Plate Number).
- ** Data Logging:** Automatically saves all detections to `traffic_data_out.csv` with timestamps.

##  Technologies Used

- **Python 3.x**
- **Ultralytics YOLOv8** (Object Detection)
- **EasyOCR** (Text Recognition)
- **OpenCV** (Image Processing)
- **Supervision** (Tracking & Counting Logic)
- **Tkinter** (Graphical User Interface)

##  Project Structure

```bash
AI-Traffic-Monitor/
├── app.py                 # Main Application Code (GUI + Logic)
├── plate_model.pt         # Custom YOLO Model for License Plates
├── yolov8n.pt             # Default YOLO Model for Vehicles
├── data/
│   └── traffic.mp4        # Test Video
├── traffic_data_log.csv   # Output Log File
└── requirements.txt       # Dependencies
