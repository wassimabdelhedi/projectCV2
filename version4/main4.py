import streamlit as st
import cv2
from yolov5 import YOLOv5
import numpy as np

# Load YOLOv8 model
model = YOLOv5("yolov8n.pt")

# Create Streamlit layout
st.title("Real-Time Object Detection with YOLOv8")
st.text("Press 'q' to exit the webcam feed")

# Define video stream
cap = cv2.VideoCapture(0)

# Function to capture frame and process it
def process_frame():
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to grab frame!")
        return None
    results = model.predict(frame)
    annotated_frame = results.render()[0]  # Get frame with bounding boxes
    return annotated_frame

# Display the webcam feed and YOLOv8 detections
while True:
    frame = process_frame()
    if frame is not None:
        # Convert frame to RGB (Streamlit expects RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)

    # Check for exit (for now, you would need to manually close Streamlit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close webcam and stop Streamlit
cap.release()
cv2.destroyAllWindows()
