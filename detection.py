from datetime import datetime, timedelta
import requests
import cv2
import numpy as np
import os
from time import sleep
from dotenv import load_dotenv
import logging
from tflite_runtime.interpreter import Interpreter

logging.basicConfig(
#    filename="/home/home/person_detect.log",
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

home_assistant_url = os.getenv("HOME_ASSISTANT_URL")
api_token = os.getenv("HOME_ASSISTANT_TOKEN")
image_frame_url = os.getenv("IMAGE_FRAME_URL")

camera_entity_ids = os.getenv("CAMERA_ENTITY_IDS").split(",")
mobile_app_ids = os.getenv("MOBILE_APP_IDS").split(",")

headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

camera_thresholds = {}

default_threshold = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
for camera_id in camera_entity_ids:
    threshold_key = f"THRESHOLD_{camera_id.replace('.', '_')}"
    camera_thresholds[camera_id] = float(os.getenv(threshold_key, default_threshold))
    logging.warning(f"Initialized threshold for {camera_id}: {camera_thresholds[camera_id]}")

IMAGE_PATH = os.getenv("IMAGE_PATH")
MAX_CONSECUTIVE_ERRORS = 3
SLEEP_INTERVAL = 1
FRAME_SLEEP = 1

CAMERAS_COUNT = len(camera_entity_ids)
CYCLE_TIME = 5
CAMERA_INTERVAL = CYCLE_TIME / CAMERAS_COUNT

# Load TFLite model
interpreter = Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()

logging.warning("Application started - Models loaded successfully")

def save_detection_image(frame, camera_entity_id, timestamp):
    """Saves detection image with timestamp."""
    filename = f"detection_{camera_entity_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(os.path.dirname(IMAGE_PATH), filename)
    cv2.imwrite(save_path, frame)
    logging.warning(f"Saved detection image: {filename}")
    return filename

def send_alarm(camera_entity_id, filename, mobile_app_id):
    """Sends an alert notification to Home Assistant mobile app."""
    notification_url = f"{home_assistant_url}/api/services/notify/{mobile_app_id}"
    data = {
        "title": "Person Detected",
        "message": f"A person has been detected on camera {camera_entity_id}",
        "data": {
            "image": f"/local/images/{filename}",
            "ttl": 0,
            "priority": "high",
            "channel": "alarm_stream"
        }
    }
    try:
        response = requests.post(notification_url, headers=headers, json=data)
        response.raise_for_status()
        logging.warning(f"Alarm sent successfully for {camera_entity_id}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending alarm for {camera_entity_id}: {e}")

def process_frame(frame, threshold):
    """Process a single frame using TFLite model."""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize frame if needed
    frame_height, frame_width = frame.shape[:2]
    if frame_height > 720 or frame_width > 1280:
        scale = min(720/frame_height, 1280/frame_width)
        frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))

    # Prepare input image
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
    detections = []
    for i in range(num_detections):
        if scores[i] > threshold:  # Use camera-specific threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame_width)
            xmax = int(xmax * frame_width)
            ymin = int(ymin * frame_height)
            ymax = int(ymax * frame_height)
            
            w = xmax - xmin
            h = ymax - ymin
            
            detection_area = w * h
            frame_area = frame_width * frame_height
            detection_ratio = detection_area / frame_area
            
            # Skip if detection is too large (over 60% of frame) or too small (under 1% of frame)
            if detection_ratio > 0.6 or detection_ratio < 0.05:
                continue
                
            detections.append((xmin, ymin, w, h, scores[i]))
            
    return frame, detections

def camera_worker(camera_entity_id):
    """Modified worker function to process quickly and return."""
    logging.warning(f"Processing camera {camera_entity_id}")
    try:
        # Take snapshot using Home Assistant camera.snapshot service
        response = requests.get(
            f"{home_assistant_url}/api/camera_proxy/{camera_entity_id}",
            headers=headers,
        )
        response.raise_for_status()
        
        # Convert response content to numpy array
        nparr = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logging.error(f"Could not decode image for {camera_entity_id}")
            return
            
        # Process the frame with camera-specific threshold
        frame, detections = process_frame(frame, camera_thresholds[camera_entity_id])
            
        if detections:
            logging.warning(f"Person Detected in Camera {camera_entity_id}")
            detection_time = datetime.now()
            frame_with_boxes = frame.copy()
            
            for x, y, w, h, score in detections:
                color = (0, 255, 0)  # Green color for detections
                cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_with_boxes, f"Person ({score:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            filename = save_detection_image(frame_with_boxes, camera_entity_id, detection_time)
            for mobile_app_id in mobile_app_ids:
                send_alarm(camera_entity_id, filename, mobile_app_id)
            
            sleep(3)

    except Exception as e:
        logging.error(f"Error processing camera {camera_entity_id}: {e}")

def check_detection_enabled():
    """Check if person detection is enabled in Home Assistant."""
    try:
        response = requests.get(
            f"{home_assistant_url}/api/states/input_boolean.person_detection",
            headers=headers
        )
        response.raise_for_status()
        state = response.json()['state']
        return state == 'on'
    except Exception as e:
        logging.error(f"Error checking person detection state: {e}")
        return True  # Default to enabled if we can't check

def main():
    """Main function to ensure each camera is checked every 5 seconds."""
    logging.warning("Starting optimized sequential person detection system")

    while True:
        if not check_detection_enabled():
            logging.warning("Person detection is disabled, sleeping for 5 seconds")
            sleep(5)
            continue
            
        cycle_start = datetime.now()
        
        for camera_id in camera_entity_ids:
            logging.warning(f"Starting camera {camera_id}")
            camera_start = datetime.now()
            camera_worker(camera_id)
            
            elapsed = (datetime.now() - camera_start).total_seconds()
            if elapsed < CAMERA_INTERVAL:
                sleep(CAMERA_INTERVAL - elapsed)
        
        cycle_elapsed = (datetime.now() - cycle_start).total_seconds()
        if cycle_elapsed < CYCLE_TIME:
            sleep(CYCLE_TIME - cycle_elapsed)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()