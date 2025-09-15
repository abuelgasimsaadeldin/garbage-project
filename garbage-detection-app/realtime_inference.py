import cv2
from ultralytics import YOLO

# Load your trained model (replace with path to your best.pt)
model = YOLO("best.pt")

# Waste classes
classes = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Open webcam (0 = default camera, change to 1 or 2 if you have multiple cams)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(source=frame, verbose=False)
    output = results[0]

    # Draw detections
    for box, conf, cls in zip(output.boxes.xyxy, output.boxes.conf, output.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_name = classes[int(cls)]
        color = (0, 255, 0)  # Green boxes for all classes

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with class name + confidence
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show result in a window
    cv2.imshow("Waste Object Detection - Real-time", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
