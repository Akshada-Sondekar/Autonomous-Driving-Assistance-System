import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load pre-trained YOLOv8 model (switch to 'yolov8m.pt' or 'yolov8l.pt' for better accuracy)
model = YOLO("yolov8m.pt")

# Define improved action logic
def take_action(detections, frame_width):
    action = "Keep Driving"
    highest_priority = None

    for obj in detections:
        label = obj['label']
        confidence = obj['confidence']
        x_center = obj['x_center']
        area = obj['area']  # Larger area means the object is closer

        if confidence > 0.6:  # Increase confidence threshold for better accuracy
            # Assign priority based on object type
            priority = 0
            if label in ["person", "bicycle", "motorcycle"]:
                priority = 3  # High danger - requires immediate action
            elif label in ["car", "truck", "bus"]:
                priority = 2  # Medium danger - requires slowing down
            elif label in ["stop sign", "traffic light"]:
                priority = 1  # Traffic rule - requires attention

            # Prioritize closer (larger) objects
            if highest_priority is None or priority > highest_priority['priority'] or (priority == highest_priority['priority'] and area > highest_priority['area']):
                highest_priority = {
                    "label": label,
                    "x_center": x_center,
                    "priority": priority,
                    "area": area
                }

    if highest_priority:
        label = highest_priority["label"]
        x_center = highest_priority["x_center"]

        # Improved decision logic based on object's position
        if label in ["person", "bicycle", "motorcycle", "car", "truck", "bus"]:
            if x_center < frame_width / 3:
                action = "Turn Right"
            elif x_center > 2 * frame_width / 3:
                action = "Turn Left"
            else:
                action = "Brake or Reverse"
        elif label in ["stop sign", "traffic light"]:
            action = "Follow Traffic Rules"

    return action

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Perform object detection with NMS to avoid duplicate detections
    results = model(frame, conf=0.6)  # Confidence threshold for accuracy
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            # Calculate object center and area
            x_center = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)  # Area to determine closeness

            detections.append({
                "label": label,
                "confidence": float(conf),
                "x_center": x_center,
                "area": area
            })

            # Draw bounding boxes with different colors based on object type
            color = (0, 255, 0)  # Green for safe objects
            if label in ["person", "bicycle", "motorcycle"]:
                color = (0, 0, 255)  # Red for immediate danger
            elif label in ["car", "truck", "bus"]:
                color = (0, 165, 255)  # Orange for vehicles

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Decision based on improved object detection
    decision = take_action(detections, frame_width)
    cv2.putText(frame, f"Decision: {decision}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Autonomous Driving System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()