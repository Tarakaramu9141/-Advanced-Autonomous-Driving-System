# src/streamlit_app.py
'''import streamlit as st
import cv2
import numpy as np
from lane_detection import LaneDetector
from object_detection import ObjectDetector
from control import VehicleController
import tempfile
import os
import asyncio
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Initialize components with error handling
    try:
        lane_detector = LaneDetector()
        object_detector = ObjectDetector()
        controller = VehicleController()
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return

    st.title("🚗 Autonomous Driving Demo")
    st.markdown("""
    ### Computer Vision Pipeline Features:
    - **Lane Detection** using Canny Edge Detection + Hough Transform
    - **Object Detection** with YOLOv5 (cars, trucks, traffic signs)
    - **Control System** with basic steering logic
    """)

    # Video input section
    uploaded_file = st.file_uploader("Upload driving video", type=["mp4", "avi"])
    temp_file = None
    use_sample = st.checkbox("Use sample video")

    # Video processing
    if st.button('Start Processing'):
        try:
            if use_sample:
                video_path = "data/videos/sample_drive.mp4"
            elif uploaded_file:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
            else:
                st.warning("Please upload a video or select sample video")
                return

            cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Lane detection
                left_lane, right_lane = lane_detector.detect_lanes(frame)
                lane_frame = lane_detector.draw_lanes(frame, left_lane, right_lane)

                # Object detection
                labels, coordinates = object_detector.detect_objects(frame)
                detection_frame = object_detector.draw_boxes(lane_frame, labels, coordinates)

                # Control system
                steering = controller.calculate_steering(left_lane, right_lane, frame.shape[1])
                speed = controller.calculate_speed(labels)

                # Add HUD overlay
                hud_text = f"Steering: {steering:.2f} | Speed: {speed} km/h | Objects: {len(labels)}"
                cv2.putText(detection_frame, hud_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display processing
                st_frame.image(detection_frame, channels="BGR")
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

            cap.release()
            if temp_file:
                os.unlink(temp_file.name)

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            if temp_file:
                os.unlink(temp_file.name)

if __name__ == "__main__":
    # Handle async for Streamlit
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    main()'''
#Includes night_vision
# src/streamlit_app.py (Enhanced Version)
import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import queue
import threading
from collections import deque
from lane_detection import NightVisionLaneDetector
from object_detection import EnhancedObjectDetector
from control import AdvancedVehicleController

# Constants
NIGHT_MODE_THRESHOLD = 50
VIDEO_FOLDER = "data/videos"
TARGET_FPS = 30

class AutonomousSystem:
    def __init__(self):
        self.lane_detector = NightVisionLaneDetector()
        self.object_detector = EnhancedObjectDetector()
        self.controller = AdvancedVehicleController()
        self.night_mode = False
        self.last_frame_time = time.time()
        self.frame_interval = 0.033  # Target 30 FPS

    def process_frame(self, frame):
        # Control frame processing rate
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return None
        self.last_frame_time = current_time

        # Night mode detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.night_mode = brightness < NIGHT_MODE_THRESHOLD

        # Night vision enhancement
        if self.night_mode:
            frame = self._enhance_night_frame(frame)

        # Process frame components
        lanes_info = self.lane_detector.detect_lanes(frame, self.night_mode)
        detections = self.object_detector.detect_objects(frame, self.night_mode)
        control_data = self.controller.process_environment(lanes_info, detections, self.night_mode)
        
        # Visualize results
        return self._visualize_frame(frame, lanes_info, detections, control_data, brightness)

    def _enhance_night_frame(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _visualize_frame(self, frame, lanes_info, detections, control_data, brightness):
        # Draw lanes
        frame_with_lanes = self.lane_detector.draw_lanes(frame, lanes_info)
        
        # Draw objects
        frame_with_objects = self.object_detector.draw_boxes(frame_with_lanes, detections)
        
        # Add HUD overlay
        self._add_hud_overlay(frame_with_objects, control_data, brightness)
        
        return frame_with_objects

    def _add_hud_overlay(self, frame, control_data, brightness):
        hud_info = [
            f"Mode: {'NIGHT' if self.night_mode else 'DAY'}",
            f"Brightness: {brightness:.1f}",
            f"Lanes: {control_data['lane_count']}",
            f"Objects: {control_data['object_count']}",
            f"Speed: {control_data['speed']} km/h",
            f"Steering: {control_data['steering']:.2f}"
        ]
        for i, text in enumerate(hud_info):
            cv2.putText(frame, text, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

class FrameProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_frames, daemon=True)
        self.system = AutonomousSystem()  # Now properly defined
        
    def start(self):
        self.thread.start()
        
    def stop(self):
        self.stop_event.set()
        self.thread.join()
        
    def _process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                processed_frame = self.system.process_frame(frame)
                if processed_frame is not None:
                    self.output_queue.put(processed_frame)
            except queue.Empty:
                continue

# ... [rest of your streamlit_app.py code continues below]

def get_video_files():
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)
        return []
    return [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

def main():
    st.set_page_config(layout="wide", page_title="Autonomous Driving Demo")
    
    # Main UI
    st.title("🚗 Advanced Autonomous Driving System")
    st.markdown("""
    **Note:** For configuration options, please check the sidebar. 
    For smoother playback, lower resolution videos work better.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        # Video source selection
        video_source = st.radio(
            "Video Source",
            options=["Upload Video", "Use Sample Videos"],
            index=0
        )
        
        video_path = None
        if video_source == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload your video file",
                type=["mp4", "avi", "mov"],
                help="For best performance, use 720p or lower resolution"
            )
            if uploaded_file:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
        else:
            video_files = get_video_files()
            if video_files:
                selected_video = st.selectbox(
                    "Select sample video",
                    video_files
                )
                video_path = os.path.join(VIDEO_FOLDER, selected_video)
            else:
                st.warning("No sample videos found in data/videos folder")
        
        # Performance settings
        st.markdown("### Performance Settings")
        target_fps = st.slider("Target FPS", 10, 60, TARGET_FPS)
        enable_hw_accel = st.checkbox("Enable Hardware Acceleration", True)
        
        # Detection settings
        st.markdown("### Detection Settings")
        detect_vehicles = st.checkbox("Vehicles", True)
        detect_pedestrians = st.checkbox("Pedestrians", True)
        detect_infrastructure = st.checkbox("Infrastructure", True)
        detect_trees = st.checkbox("Trees/Vegetation", True)

    # Main processing
    if st.button('Start Processing'):
        if not video_path:
            st.warning("Please select or upload a video first")
            return
        
        frame_processor = FrameProcessor()
        frame_processor.start()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                return
            
            st_frame = st.empty()
            fps_display = st.empty()
            perf_stats = st.empty()
            
            last_display_time = time.time()
            frame_count = 0
            fps_history = deque(maxlen=10)
            processing_times = deque(maxlen=10)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Put frame in processing queue
                try:
                    frame_processor.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    continue
                
                # Get processed frames if available
                try:
                    start_time = time.time()
                    processed_frame = frame_processor.output_queue.get(timeout=0.1)
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # Adaptive display rate
                    current_time = time.time()
                    frame_count += 1
                    
                    if current_time - last_display_time >= 1.0/target_fps:
                        st_frame.image(processed_frame, channels="BGR")
                        
                        # Calculate and display metrics
                        fps = 1 / (current_time - last_display_time)
                        fps_history.append(fps)
                        
                        fps_display.text(f"Display FPS: {np.mean(fps_history):.1f}")
                        perf_stats.text(
                            f"Processing: {1/np.mean(processing_times):.1f} FPS | "
                            f"Latency: {np.mean(processing_times)*1000:.1f}ms | "
                            f"Queue: {frame_processor.output_queue.qsize()}"
                        )
                        
                        last_display_time = current_time
                        
                except queue.Empty:
                    continue
                    
        finally:
            frame_processor.stop()
            cap.release()
            if 'temp_file' in locals():
                os.unlink(temp_file.name)

if __name__ == "__main__":
    main()