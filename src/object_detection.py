'''import cv2
import torch
import numpy as np

class ObjectDetector:
    def __init__(self):
        # Load YOLOv5 with error handling
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = 0.5  # Confidence threshold
            self.classes = self.model.names
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_objects(self, frame):
        try:
            # Convert frame to RGB and predict
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb)
            
            # Extract detections with proper error handling
            detections = results.pred[0].cpu().numpy()
            
            if len(detections) == 0:
                return np.array([]), np.array([])
                
            # Split into coordinates and labels
            coordinates = detections[:, :4]  # x1, y1, x2, y2
            labels = detections[:, 5]       # class IDs
            
            return labels, coordinates
            
        except Exception as e:
            print(f"Detection error: {e}")
            return np.array([]), np.array([])

    def draw_boxes(self, frame, labels, coordinates):
        if len(labels) == 0:
            return frame
            
        height, width = frame.shape[:2]
        relevant_classes = [2, 3, 5, 7, 9]  # car, motorcycle, bus, truck, traffic light
        
        for i in range(len(labels)):
            if labels[i] in relevant_classes:
                x1, y1, x2, y2 = map(int, coordinates[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, self.classes[int(labels[i])], 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        return frame'''
#Includes night vision
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO

class EnhancedObjectDetector:
    def __init__(self):
        # Enable GPU optimizations
        cudnn.benchmark = True
        torch.set_flush_denormal(True)
        
        # Load optimized model
        self.model = YOLO('yolov8s.pt').half().eval()  # Half precision
        self.model.fuse()
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Detection categories
        self.categories = {
            'vehicle': {
                'classes': [2, 3, 5, 7, 8, 9],
                'color': (0, 255, 0)
            },
            'person': {
                'classes': [0, 1],
                'color': (255, 0, 0)
            },
            'traffic': {
                'classes': [9, 11, 13],
                'color': (0, 255, 255)
            },
            'vegetation': {
                'classes': [59, 56],
                'color': (100, 255, 100)
            },
            'structure': {
                'classes': [57, 58, 60, 61, 62, 63],
                'color': (0, 0, 255)
            }
        }
        
        # Confidence thresholds
        self.day_conf = 0.3
        self.night_conf = 0.2
    
    def detect_objects(self, frame, night_mode=False):
        try:
            # Preprocess - resize for faster inference
            resized = cv2.resize(frame, (640, 384))
            
            with torch.no_grad():  # Disable gradients for inference
                results = self.model(
                    resized,
                    verbose=False,
                    imgsz=640,
                    conf=self.night_conf if night_mode else self.day_conf
                )
            
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confs, clss):
                    for cat_name, cat_data in self.categories.items():
                        if cls in cat_data['classes']:
                            detections.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'category': cat_name,
                                'color': cat_data['color']
                            })
                            break
            
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def draw_boxes(self, frame, detections):
        if not detections:
            return frame
            
        height, width = frame.shape[:2]
        
        for det in detections:
            try:
                box = det['box']
                # Scale coordinates back to original size
                x_scale = width / 640
                y_scale = height / 384
                
                x1, y1, x2, y2 = [
                    int(box[0] * x_scale),
                    int(box[1] * y_scale),
                    int(box[2] * x_scale),
                    int(box[3] * y_scale)
                ]
                
                # Validate coordinates
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1))
                y2 = max(0, min(y2, height-1))
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), det['color'], 2)
                
                # Draw label
                label = f"{self.model.names[det['cls']]} {det['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, det['color'], 1)
            except Exception as e:
                print(f"Error drawing box: {e}")
                continue
                
        return frame