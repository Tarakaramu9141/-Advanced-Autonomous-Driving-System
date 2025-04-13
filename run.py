from src.streamlit_app import main

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "local":
        # Local testing without Streamlit
        from src.lane_detection import LaneDetector
        from src.object_detection import ObjectDetector
        import cv2
        
        cap = cv2.VideoCapture("data/videos/sample_drive.mp4")
        lane_detector = LaneDetector()
        object_detector = ObjectDetector()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            left_lane, right_lane = lane_detector.detect_lanes(frame)
            frame_with_lanes = lane_detector.draw_lanes(frame, left_lane, right_lane)
            
            labels, coordinates = object_detector.detect_objects(frame)
            frame_with_objects = object_detector.draw_boxes(frame_with_lanes, labels, coordinates)
            
            cv2.imshow("Autonomous Driving", frame_with_objects)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Run Streamlit app
        main()