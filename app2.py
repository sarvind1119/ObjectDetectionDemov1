from ultralytics import YOLO
import cv2
import math
import threading
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Change this line in your code
model = YOLO(resource_path("yolo-Weights/yolov8x.pt"))

# Fetch model's built-in class names
classNames = model.names

# Camera URLs
camera_urls = {
    1: "rtsp://admin:Nimda@2024@10.10.116.70:554/media/video1",
    2: "rtsp://admin:Nimda@2024@10.10.116.71:554/media/video1",
    3: "rtsp://admin:Nimda@2024@10.10.116.72:554/media/video1",
    4: "rtsp://admin:Nimda@2024@10.10.116.73:554/media/video1",
    5: "rtsp://admin:Nimda@2024@10.10.116.74:554/media/video1",
    6: "rtsp://admin:Nimda@2024@10.10.116.75:554/media/video1"
}

# Function for object detection
def object_detection(cap, window_name):
    while True:
        success, img = cap.read()
        if not success:
            print(f"Error: Unable to fetch the frame from {window_name}")
            break

        results = model(img, stream=True)

        # Draw bounding boxes and class labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Class name and confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls] if cls < len(classNames) else "Unknown"

                # Display class name and confidence
                text = f"{class_name} ({confidence})"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the video frame with detections
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Function for handling all cameras
def handle_all_cameras():
    threads = []
    for camera_id, rtsp_url in camera_urls.items():
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            thread = threading.Thread(target=object_detection, args=(cap, f"Camera {camera_id}"))
            threads.append(thread)
            thread.start()
        else:
            print(f"Error: Unable to access feed from Camera {camera_id}")

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def main():
    while True:
        print("Object detection on webcam or live feed of classroom?")
        print("Press 1 for webcam or 2 for classroom feed.")

        user_input = input("Enter your choice: ")

        if user_input == "1":
            print("Starting object detection on webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Unable to access webcam.")
                return
            object_detection(cap, "Webcam")
            break

        elif user_input == "2":
            while True:
                print("Do you want the live feed of all cameras or a specific camera?")
                print("Press 1 for all cameras or 2 for a specific camera.")

                feed_input = input("Enter your choice: ")

                if feed_input == "1":
                    print("Starting object detection on all cameras...")
                    handle_all_cameras()
                    break
                elif feed_input == "2":
                    print("Available cameras:")
                    for camera_id in camera_urls.keys():
                        print(f"Camera {camera_id}")

                    camera_choice = input("Enter the camera number you want to view: ")

                    if camera_choice.isdigit() and int(camera_choice) in camera_urls:
                        camera_id = int(camera_choice)
                        print(f"Starting object detection on Camera {camera_id}...")
                        cap = cv2.VideoCapture(camera_urls[camera_id])
                        if not cap.isOpened():
                            print(f"Error: Unable to access feed from Camera {camera_id}")
                            return
                        object_detection(cap, f"Camera {camera_id}")
                        break
                    else:
                        print("Invalid camera number. Please enter a valid camera number.")
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()

