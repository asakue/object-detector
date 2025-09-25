import cv2
import numpy as np
import os
from ultralytics import YOLO
import time

class OptimizedDetector:
    def __init__(self, unknown_threshold=0.3):
        self.unknown_threshold = unknown_threshold
        self.model_cache = {}
        
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_ready = True
            dummy_input = np.random.rand(640, 640, 3).astype(np.uint8)
            _ = self.yolo_model(dummy_input, verbose=False)
        except:
            self.yolo_ready = False
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.opencv_ready = True
        except:
            self.opencv_ready = False
        
        self.distance_calibration = {
            'person': 40, 'car': 180, 'chair': 50, 'cup': 10, 
            'laptop': 35, 'cell phone': 8, 'book': 20, 'bottle': 8, 'default': 30
        }
        
        self.focal_length = 1000
        self.colors = {
            'high_confidence': (0, 255, 0), 'medium_confidence': (0, 200, 255), 
            'low_confidence': (0, 100, 255), 'unknown': (128, 0, 128),
            'face': (255, 0, 0), 'eye': (0, 0, 255), 'distance': (255, 255, 0)
        }
        
        self.previous_detections = []
        self.frame_count = 0
        self.processing_times = []
        self.last_time = time.time()
        self.fps = 0

    def calculate_distance(self, bbox, object_class, frame_width):
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        real_width = self.distance_calibration.get(object_class, self.distance_calibration['default'])
        if pixel_width > 0:
            distance_cm = (real_width * self.focal_length) / pixel_width
            distance_m = distance_cm / 100
            return max(0.1, distance_m)
        return None

    def detect_yolo_objects(self, frame):
        if not self.yolo_ready:
            return []
        
        try:
            start_time = time.time()
            results = self.yolo_model(frame, conf=0.25, iou=0.45, verbose=False, imgsz=640)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area < 500:
                        continue
                    
                    if confidence < self.unknown_threshold:
                        det_type = 'unknown'
                        display_class = 'Unknown'
                        color = self.colors['unknown']
                    elif confidence < 0.5:
                        det_type = 'low_confidence'
                        display_class = f"{class_name}?"
                        color = self.colors['low_confidence']
                    elif confidence < 0.7:
                        det_type = 'medium_confidence'
                        display_class = class_name
                        color = self.colors['medium_confidence']
                    else:
                        det_type = 'high_confidence'
                        display_class = class_name
                        color = self.colors['high_confidence']
                    
                    distance = self.calculate_distance((x1, y1, x2, y2), class_name, frame.shape[1])
                    
                    detections.append({
                        'type': det_type, 'class': display_class, 'confidence': confidence,
                        'bbox': (x1, y1, x2, y2), 'color': color, 'original_class': class_name,
                        'distance': distance, 'area': bbox_area
                    })
            
            return detections
        except:
            return []

    def detect_opencv_objects(self, frame):
        if not self.opencv_ready:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            detections = []
            for (x, y, w, h) in faces:
                distance = self.calculate_distance((x, y, x+w, y+h), 'person', frame.shape[1])
                
                detections.append({
                    'type': 'face', 'class': 'Face', 'confidence': 0.9,
                    'bbox': (x, y, x + w, y + h), 'color': self.colors['face'],
                    'original_class': 'Face', 'distance': distance, 'area': w * h
                })
                
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
                for (ex, ey, ew, eh) in eyes:
                    eye_distance = self.calculate_distance((x+ex, y+ey, x+ex+ew, y+ey+eh), 'default', frame.shape[1])
                    detections.append({
                        'type': 'eye', 'class': 'Eye', 'confidence': 0.8,
                        'bbox': (x + ex, y + ey, x + ex + ew, y + ey + eh), 
                        'color': self.colors['eye'], 'original_class': 'Eye',
                        'distance': eye_distance, 'area': ew * eh
                    })
            
            return detections
        except:
            return []

    def track_objects(self, current_detections, previous_detections):
        if not previous_detections:
            return current_detections
        
        tracked_detections = []
        used_indices = set()
        
        for prev_det in previous_detections:
            best_match = None
            best_iou = 0.3
            
            for i, curr_det in enumerate(current_detections):
                if i in used_indices:
                    continue
                
                iou = self.calculate_iou(prev_det['bbox'], curr_det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = (i, curr_det)
            
            if best_match:
                i, curr_det = best_match
                smoothed_bbox = self.smooth_bbox(prev_det['bbox'], curr_det['bbox'])
                curr_det['bbox'] = smoothed_bbox
                curr_det['tracked'] = True
                tracked_detections.append(curr_det)
                used_indices.add(i)
        
        for i, det in enumerate(current_detections):
            if i not in used_indices:
                det['tracked'] = False
                tracked_detections.append(det)
        
        return tracked_detections

    def smooth_bbox(self, prev_bbox, curr_bbox, alpha=0.7):
        x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
        x1_curr, y1_curr, x2_curr, y2_curr = curr_bbox
        
        x1 = int(alpha * x1_prev + (1 - alpha) * x1_curr)
        y1 = int(alpha * y1_prev + (1 - alpha) * y1_curr)
        x2 = int(alpha * x2_prev + (1 - alpha) * x2_curr)
        y2 = int(alpha * y2_prev + (1 - alpha) * y2_curr)
        
        return (x1, y1, x2, y2)

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def filter_detections(self, detections):
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            
            if width < 20 or height < 20 or width > 1000 or height > 1000:
                continue
            
            aspect_ratio = width / height
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            filtered.append(det)
        return filtered

    def draw_detections(self, frame, detections):
        result_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance')
            
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))
            
            thickness = 3 if confidence > 0.7 else 2
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            info_text = f"{class_name} {confidence:.2f}"
            if distance is not None:
                info_text += f" {distance:.1f}m"
            
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            text_x = x1
            text_y = y1 - 10
            
            if text_y < 30:
                text_y = y2 + text_size[1] + 10
            
            text_x = max(0, min(text_x, frame_width - text_size[0] - 5))
            text_y = max(30, min(text_y, frame_height - 10))
            
            cv2.rectangle(result_frame, (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0], text_y + 5), color, -1)
            
            cv2.putText(result_frame, info_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame

def main():
    try:
        detector = OptimizedDetector(unknown_threshold=0.3)
    except:
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        return
    
    settings = {'show_tracking': True}
    frame_count = 0
    last_fps_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        if current_time - last_fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_fps_time = current_time
        
        yolo_detections = detector.detect_yolo_objects(frame)
        opencv_detections = detector.detect_opencv_objects(frame)
        all_detections = yolo_detections + opencv_detections
        
        filtered_detections = detector.filter_detections(all_detections)
        
        if settings['show_tracking'] and detector.previous_detections:
            filtered_detections = detector.track_objects(filtered_detections, detector.previous_detections)
        
        result_frame = detector.draw_detections(frame, filtered_detections)
        
        cv2.putText(result_frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Objects: {len(filtered_detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        detector.previous_detections = filtered_detections
        
        cv2.imshow('Object Detector - Press Q to quit', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, result_frame)
        elif key == ord('t'):
            settings['show_tracking'] = not settings['show_tracking']
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()