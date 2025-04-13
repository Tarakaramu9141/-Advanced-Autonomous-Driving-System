import numpy as np

class AdvancedVehicleController:
    def __init__(self):
        self.steering_angle = 0
        self.speed = 0
        self.lane_change_threshold = 0.3
        self.night_speed_reduction = 0.7  # Reduce speed by 30% at night
    
    def process_environment(self, lanes_info, detections, night_mode=False):
        """Process all environment inputs and return control commands"""
        control_data = {
            'steering': self._calculate_steering(lanes_info),
            'speed': self._calculate_speed(detections, night_mode),
            'lane_count': lanes_info.get('lane_count', 1),
            'object_count': len(detections)
        }
        return control_data
    
    def _calculate_steering(self, lanes_info):
        """Enhanced steering calculation with lane keeping"""
        left_lane = lanes_info.get('left_lane')
        right_lane = lanes_info.get('right_lane')
        
        if left_lane is None or right_lane is None:
            return 0  # No steering if lanes not detected
        
        # Calculate lane centers
        left_center = (left_lane[0] + left_lane[2]) / 2
        right_center = (right_lane[0] + right_lane[2]) / 2
        lane_center = (left_center + right_center) / 2
        
        # Calculate deviation from center
        steering = (lane_center - 320) / 320  # Normalized to -1 to 1
        
        # Smooth steering with deadzone
        if abs(steering) < 0.1:
            return 0
        return np.clip(steering, -1, 1)
    
    def _calculate_speed(self, detections, night_mode=False):
        """Dynamic speed control based on environment"""
        base_speed = 30 if night_mode else 50
        min_distance = self._get_min_distance(detections)
        
        if min_distance < 50:  # Very close object
            return 0
        elif min_distance < 100:  # Close object
            return base_speed * 0.5
        elif night_mode:
            return base_speed * self.night_speed_reduction
        return base_speed
    
    def _get_min_distance(self, detections):
        """Calculate distance to nearest relevant object"""
        if len(detections) == 0:
            return float('inf')
        
        # Get closest object (using bottom center as reference)
        distances = []
        for det in detections:
            try:
                box = det['box']
                x1, y1, x2, y2 = box[:4]
                
                obj_bottom_center = ((x1+x2)/2, y2)
                distance = np.sqrt((320 - obj_bottom_center[0])**2 + 
                                 (480 - obj_bottom_center[1])**2)
                distances.append(distance)
            except Exception as e:
                print(f"Error calculating distance: {e}")
                continue
        
        return min(distances) if distances else float('inf')