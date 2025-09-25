# Object Detector

Real-time object detection application with distance measurement using webcam.

##  Features

- Real-time object detection (80+ classes)
- Face and eye detection
- Distance estimation to objects
- Object tracking between frames
- Unknown object detection
- FPS counter and performance stats

##  Technologies Used

- **Python** - Main programming language
- **OpenCV** - Computer vision and camera processing
- **YOLOv8** - Deep learning object detection
- **Ultralytics** - YOLO model integration
- **Haar Cascades** - Face and eye detection

##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/object-detector.git
cd object-detector
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python detector.py
```
# Detection Capabilities
## YOLO Objects (80+ classes):
People, animals, vehicles

Household items, electronics

Food, sports equipment

## OpenCV Detection:
Faces

Eyes (within detected faces)

## Output Information
For each detected object:

Object name and confidence level

Bounding box coordinates

Distance from camera (in meters)

## Requirements
Python 3.7+

Webcam

2GB+ RAM

Windows/Linux/macOS
