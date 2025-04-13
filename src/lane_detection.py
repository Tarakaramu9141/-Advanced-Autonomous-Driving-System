'''import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        self.left_lane = []
        self.right_lane = []
        
    def detect_lanes(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Create mask for region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height * 1 // 2),
            (width, height * 1 // 2),
            (width, height),
            (0, height),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, 
                               np.array([]), minLineLength=40, maxLineGap=5)
        
        # Separate left and right lines
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:  # Vertical line, skip
                    continue
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))
        
        # Average lines with proper checks
        left_avg = np.average(left_lines, axis=0) if len(left_lines) > 0 else None
        right_avg = np.average(right_lines, axis=0) if len(right_lines) > 0 else None
        
        # Calculate lane lines with proper None checks
        left_lane = self.make_points(frame, left_avg) if left_avg is not None else None
        right_lane = self.make_points(frame, right_avg) if right_avg is not None else None
        
        return left_lane, right_lane
    
    def make_points(self, frame, line):
        height, width = frame.shape[:2]
        slope, intercept = line
        y1 = height
        y2 = int(y1 * 1 / 2)
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]
    
    def draw_lanes(self, frame, left_lane, right_lane):
        line_image = np.zeros_like(frame)
        
        if left_lane is not None:
            for line in left_lane:
                cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 10)
        
        if right_lane is not None:
            for line in right_lane:
                cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 10)
        
        return cv2.addWeighted(frame, 0.8, line_image, 1, 1)'''
#includes night vision
import cv2
import numpy as np
from collections import deque

class NightVisionLaneDetector:
    def __init__(self):
        self.cache_size = 5
        self.lane_cache = deque(maxlen=self.cache_size)
        self.last_frame = None
        
    def detect_lanes(self, frame, night_mode=False):
        # Return cached result if frame is similar
        if self.last_frame is not None and self._frames_similar(frame, self.last_frame):
            if len(self.lane_cache) > 0:
                return self.lane_cache[-1]
        
        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if night_mode:
            # Night vision enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            gray = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)[:,:,0]
        
        # Edge detection with adaptive thresholds
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # ROI mask
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (width*0.1, height*0.9),
            (width*0.9, height*0.9),
            (width*0.6, height*0.6),
            (width*0.4, height*0.6)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
        
        # Process lines
        left_lines, right_lines = self._process_lines(lines, width)
        
        result = {
            'left_lane': self._average_lines(left_lines, frame.shape),
            'right_lane': self._average_lines(right_lines, frame.shape),
            'lane_count': min(len(left_lines), len(right_lines))
        }
        
        # Cache results
        self.lane_cache.append(result)
        self.last_frame = frame.copy()
        return result
    
    def _frames_similar(self, frame1, frame2, threshold=0.9):
        # Simple similarity check using histogram comparison
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0,256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) > threshold
    
    def _process_lines(self, lines, img_width):
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                
                slope = (y2-y1)/(x2-x1)
                if abs(slope) < 0.5:
                    continue
                
                if slope < 0 and x1 < img_width/2:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > img_width/2:
                    right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def _average_lines(self, lines, frame_shape):
        if not lines:
            return None
            
        avg_line = np.mean(lines, axis=0)
        height, width = frame_shape[:2]
        
        # Convert to two points
        x1, y1, x2, y2 = avg_line
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        
        # Calculate points at bottom and middle of frame
        y1 = height
        y2 = int(height*0.6)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        
        return [x1, y1, x2, y2]
    
    def draw_lanes(self, frame, lanes_info):
        line_img = np.zeros_like(frame)
        
        if lanes_info['left_lane'] is not None:
            x1, y1, x2, y2 = lanes_info['left_lane']
            cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 10)
        
        if lanes_info['right_lane'] is not None:
            x1, y1, x2, y2 = lanes_info['right_lane']
            cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 10)
        
        # Add lane count
        cv2.putText(line_img, f"Lanes: {lanes_info['lane_count']}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        return cv2.addWeighted(frame, 0.8, line_img, 1, 0)