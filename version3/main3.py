from yolov5 import YOLOv5
import cv2
from collections import Counter

# Load the YOLOv5 model (using the strongest one, e.g., yolov5x)
model = YOLOv5("yolov5x.pt")  # 'yolov5x.pt' is the strongest variant

# Start the webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the current frame
    results = model.predict(frame)  # Correct usage of predict method

    detected_classes = []

    # Loop through the results to get the class names and bounding box details
    for result in results.xyxy[0]:  # xyxy are the bounding box coordinates
        cls = int(result[5])  # Class ID (index)
        class_name = results.names[cls]  # Access class names from results
        detected_classes.append(class_name)

    # Count the occurrences of each detected object
    counter = Counter(detected_classes)

    # Generate a comment based on detected objects
    if len(counter) == 0:
        comment = "Searching for objects..."
    else:
        comment = "Detected: " + ", ".join([f"{obj}({count})" for obj, count in counter.items()])

    # Annotate the frame with bounding boxes and class names
    annotated_frame = results.render()[0]  # This will add bounding boxes and labels

    # Create a copy of the annotated frame to make it writable
    annotated_frame_copy = annotated_frame.copy()

    # Add the comment text to the frame
    cv2.putText(annotated_frame_copy, comment, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the result in real-time
    cv2.imshow("Real-Time Object Detection", annotated_frame_copy)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
