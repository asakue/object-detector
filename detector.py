import cv2
import numpy as np
import os
from ultralytics import YOLO
import time

class OptimizedDetector:
    def __init__(self, unknown_threshold=0.3):
        """
        детектор с расчетом расстояния
        """
        print("🔄 Инициализация детектора...")
        
        self.unknown_threshold = unknown_threshold
        
        self.model_cache = {}
        
        try:
            self.yolo_model = YOLO('yolov8m.pt') 
            self.yolo_ready = True
            
            dummy_input = np.random.rand(640, 640, 3).astype(np.uint8)
            _ = self.yolo_model(dummy_input, verbose=False)
            print("✅ YOLO детектор загружен")
        except Exception as e:
            print(f"❌ YOLO не доступен: {e}")
            try:
                # Fallback на nano модель
                self.yolo_model = YOLO('yolov8n.pt')
                self.yolo_ready = True
                print("✅ YOLO nano модель загружена")
            except:
                self.yolo_ready = False
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.opencv_ready = True
            print("✅ OpenCV детекторы загружены")
        except Exception as e:
            print(f"❌ OpenCV детекторы не доступны: {e}")
            self.opencv_ready = False
        
        self.distance_calibration = {
            'person': 40,      
            'car': 180,      
            'chair': 50,       
            'cup': 10,         
            'laptop': 35,     
            'cell phone': 8,  
            'book': 20,        
            'bottle': 8,      
            'default': 30     
        }
        
        self.focal_length = 1000 
        
        self.colors = {
            'high_confidence': (0, 255, 0),     
            'medium_confidence': (0, 200, 255), 
            'low_confidence': (0, 100, 255),    
            'unknown': (128, 0, 128),          
            'face': (255, 0, 0),               
            'eye': (0, 0, 255),                 
            'distance': (255, 255, 0)           
        }
        
        self.previous_detections = []
        self.frame_count = 0
        
        self.processing_times = []
        
        print("✅ Детектор инициализирован с оптимизациями")
    
    def calculate_distance(self, bbox, object_class, frame_width):
        """
        Расчет расстояния до объекта на основе его размера в кадре
        Формула: distance = (real_width * focal_length) / pixel_width
        """
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        
        real_width = self.distance_calibration.get(object_class, 
                         self.distance_calibration['default'])
        
        if pixel_width > 0:
            distance_cm = (real_width * self.focal_length) / pixel_width
            distance_m = distance_cm / 100
            return max(0.1, distance_m)  
        return None
    
    def auto_calibrate_focal_length(self, known_objects):
        """
        Автоматическая калибровка фокусного расстояния на основе известных объектов
        """
        if len(known_objects) < 3:
            return self.focal_length
        
        try:
            calibrations = []
            for obj in known_objects:
                if obj['confidence'] > 0.7: 
                    class_name = obj['original_class']
                    real_width = self.distance_calibration.get(class_name)
                    if real_width:
                        x1, y1, x2, y2 = obj['bbox']
                        pixel_width = x2 - x1
                        assumed_distance = 200  # см
                        focal = (pixel_width * assumed_distance) / real_width
                        calibrations.append(focal)
            
            if calibrations:
                self.focal_length = np.median(calibrations)
                self.focal_length = max(500, min(2000, self.focal_length))
        
        except Exception as e:
            print(f"Ошибка калибровки: {e}")
        
        return self.focal_length
    
    def detect_yolo_objects(self, frame):
        """Оптимизированное детектирование объектов с помощью YOLO"""
        if not self.yolo_ready:
            return []
        
        try:
            start_time = time.time()
            
            results = self.yolo_model(frame, 
                                    conf=0.25,  
                                    iou=0.45,   
                                    verbose=False,
                                    imgsz=640)  
            
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
                        display_class = 'Unknown Object'
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
                        'type': det_type,
                        'class': display_class,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'color': color,
                        'original_class': class_name,
                        'distance': distance,
                        'area': bbox_area
                    })
            
            return detections
        except Exception as e:
            print(f"Ошибка YOLO детекции: {e}")
            return []
    
    def detect_opencv_objects(self, frame):
        """Оптимизированное детектирование лиц"""
        if not self.opencv_ready:
            return []
        
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  
                minNeighbors=6,    
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            for (x, y, w, h) in faces:
                x, y, w, h = x*2, y*2, w*2, h*2
                
                distance = self.calculate_distance((x, y, x+w, y+h), 'person', frame.shape[1])
                
                detections.append({
                    'type': 'face',
                    'class': 'Face',
                    'confidence': 0.9,
                    'bbox': (x, y, x + w, y + h),
                    'color': self.colors['face'],
                    'original_class': 'Face',
                    'distance': distance,
                    'area': w * h
                })
            
            return detections
        except Exception as e:
            print(f"Ошибка OpenCV детекции: {e}")
            return []
    
    def track_objects(self, current_detections, previous_detections):
        """Простой трекинг объектов между кадрами для стабильности"""
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
        """Сглаживание bounding box для уменьшения дрожания"""
        x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
        x1_curr, y1_curr, x2_curr, y2_curr = curr_bbox
        
        x1 = int(alpha * x1_prev + (1 - alpha) * x1_curr)
        y1 = int(alpha * y1_prev + (1 - alpha) * y1_curr)
        x2 = int(alpha * x2_prev + (1 - alpha) * x2_curr)
        y2 = int(alpha * y2_prev + (1 - alpha) * y2_curr)
        
        return (x1, y1, x2, y2)
    
    def calculate_iou(self, box1, box2):
        """Быстрое вычисление IoU"""
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
        """Фильтрация обнаружений по качеству"""
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
        """отрисовка с информацией о расстоянии"""
        result_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance')
            is_tracked = det.get('tracked', False)
            
            thickness = 3 if confidence > 0.7 else 2
            if is_tracked:
                thickness += 1 
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            main_text = f"{class_name} ({confidence:.2f})"
            if distance is not None:
                distance_text = f"{distance:.1f}m"
            else:
                distance_text = "?m"
            
            main_text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            distance_text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(result_frame, (x1, y1 - main_text_size[1] - 10),
                         (x1 + main_text_size[0], y1), color, -1)
            
            cv2.putText(result_frame, main_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.rectangle(result_frame, (x1, y2),
                         (x1 + distance_text_size[0], y2 + distance_text_size[1] + 10),
                         self.colors['distance'], -1)
            
            cv2.putText(result_frame, distance_text, (x1, y2 + distance_text_size[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(result_frame, (center_x, center_y), 3, color, -1)
        
        return result_frame

def main():
    try:
        detector = OptimizedDetector(unknown_threshold=0.3)
        print("✅ Оптимизированный детектор готов!")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
    
    if not cap.isOpened():
        print("❌ Не удалось открыть камеру")
        return
    
    print("\n🎯 Оптимизированный детектор запущен!")
    print("📊 Возможности:")
    print("   • Высокоточное распознавание объектов")
    print("   • Расчет расстояния до объектов")
    print("   • Трекинг объектов между кадрами")
    print("   • Автоматическая калибровка")
    print("\n🎮 Управление:")
    print("   • Q - выход")
    print("   • S - сохранить снимок")
    print("   • 1-5 - настройка чувствительности")
    print("   • C - калибровка расстояния")
    print("   • T - вкл/выкл трекинг")
    
    settings = {
        'show_tracking': True,
        'auto_calibrate': True,
        'min_confidence': 0.3
    }
    
    frame_count = 0
    fps = 0
    start_time = time.time()
    previous_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 2 == 0 and fps < 15:
            continue

        yolo_detections = detector.detect_yolo_objects(frame)
        opencv_detections = detector.detect_opencv_objects(frame)
        all_detections = yolo_detections + opencv_detections
        
        filtered_detections = detector.filter_detections(all_detections)

        if settings['show_tracking'] and previous_detections:
            filtered_detections = detector.track_objects(filtered_detections, previous_detections)

        if settings['auto_calibrate'] and frame_count % 30 == 0:
            known_objects = [d for d in filtered_detections if d['confidence'] > 0.7]
            detector.auto_calibrate_focal_length(known_objects)

        result_frame = detector.draw_detections(frame, filtered_detections)

        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - start_time)
            start_time = current_time

        avg_processing_time = np.mean(detector.processing_times) if detector.processing_times else 0
        stats = [
            f"FPS: {fps:.1f}",
            f"Processing: {avg_processing_time*1000:.1f}ms",
            f"Objects: {len(filtered_detections)}",
            f"Focal: {detector.focal_length:.0f}px",
            f"Tracked: {len([d for d in filtered_detections if d.get('tracked')])}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(result_frame, stat, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        legend_items = [
            ("High conf", detector.colors['high_confidence']),
            ("Medium", detector.colors['medium_confidence']),
            ("Low", detector.colors['low_confidence']),
            ("Unknown", detector.colors['unknown']),
            ("Distance", detector.colors['distance'])
        ]
        
        for i, (text, color) in enumerate(legend_items):
            cv2.putText(result_frame, text, (result_frame.shape[1] - 150, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Optimized Object Detector with Distance - Press Q', result_frame)

        previous_detections = filtered_detections

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"optimized_detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"📸 Снимок сохранен: {filename}")
        elif key == ord('t'):
            settings['show_tracking'] = not settings['show_tracking']
            print(f"🔧 Трекинг: {'ВКЛ' if settings['show_tracking'] else 'ВЫКЛ'}")
        elif key == ord('c'):
            settings['auto_calibrate'] = not settings['auto_calibrate']
            print(f"🔧 Автокалибровка: {'ВКЛ' if settings['auto_calibrate'] else 'ВЫКЛ'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Программа завершена")

if __name__ == "__main__":
    main()