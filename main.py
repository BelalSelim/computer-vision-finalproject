import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # SORT tracker for object tracking
import time
import socket

# Load the YOLOv8n object detection model
model = YOLO("yolov8n.pt")

# Open a video stream (from an IP camera or mobile IP stream)
cap = cv2.VideoCapture('http://192.168.185.30:8000/stream.mjpg')

# Initialize the SORT tracker
tracker = Sort()

# Sets to store unique IDs of detected persons and cars
person_ids = set()
car_ids = set()

# Variables for timing and alert control
last_reset_time = time.time()
alert_sent = False

# X-coordinate of the vertical line for counting
line_x = 320

# Function to send a signal to a Raspberry Pi via socket when a person count threshold is reached
def notify_raspberry():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("192.168.185.30", 9999))  # Connect to Raspberry Pi server
        s.sendall(b"3")  # Send the alert code
        s.close()
        print(" Sent '3' to Raspberry Pi")
    except Exception as e:
        print(" Socket Error:", e)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model inference on the current frame
    results = model(frame, verbose=False)[0]

    # Prepare detection results for the tracker
    detections = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        class_id = int(cls)
        if class_id in [0, 2]:  # 0 = person, 2 = car (based on COCO classes)
            detections.append([x1, y1, x2, y2, 0.9])  # Confidence set to 0.9

    # Convert detections to NumPy array
    detections_np = np.array(detections, dtype=np.float32)
    if detections_np.shape[0] == 0:
        detections_np = np.empty((0, 5), dtype=np.float32)

    # Update the tracker with the current detections
    tracks = tracker.update(detections_np)

    # Draw the vertical line on the frame for reference
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)

    # Loop over tracked objects
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cx = int((x1 + x2) / 2)  # Center X of the bounding box
        cy = int((y1 + y2) / 2)  # Center Y of the bounding box
        class_name = "Unknown"

        # Match tracking box with YOLO detection class
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            bx1, by1, bx2, by2 = map(int, box)
            if abs(x1 - bx1) < 20 and abs(y1 - by1) < 20:
                if int(cls) == 0:  # Person
                    class_name = "Person"
                    if cx > line_x and track_id not in person_ids:
                        person_ids.add(track_id)
                        print(f"Person {track_id} crossed the line")
                elif int(cls) == 2:  # Car
                    class_name = "Car"
                    car_ids.add(track_id)
                break

        # Draw the tracked object and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
        cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # Display the count of unique detected persons and cars
    cv2.putText(frame, f"Unique Persons: {len(person_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Unique Cars: {len(car_ids)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Send alert to Raspberry Pi if more than 3 unique persons detected
    if len(person_ids) > 3 and not alert_sent:
        notify_raspberry()
        alert_sent = True

    # Reset counters and alert every 20 seconds
    if time.time() - last_reset_time >= 20:
        person_ids.clear()
        car_ids.clear()
        alert_sent = False
        last_reset_time = time.time()
        print("Reset counters")

    # Show the output frame
    cv2.imshow("YOLO + Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
