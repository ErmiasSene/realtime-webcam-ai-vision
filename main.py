import cv2
from ultralytics import YOLO

# Load lightweight YOLOv8 nano model (downloads automatically on first run)
model = YOLO('yolov8n.pt')

# Open laptop webcam (0 corresponds to built-in camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Camera feed unreadable.")
        break

    # Run AI detection on current frame
    results = model(frame, stream=True)

    # Draw bounding boxes and labels
    for result in results:
        annotated_frame = result.plot()

    # Display live feed
    cv2.imshow('AI Vision Prototype', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()