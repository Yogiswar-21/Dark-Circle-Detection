import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('/home/rguktrkvalley/Documents/Mini project/best2.pt')

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture image and detect dark circles.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show live webcam feed
    cv2.imshow("Webcam - Press 'c' to capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Image captured! Running detection...")

        # Run YOLO model on the captured frame
        results = model(frame)
        predictions = results[0].boxes.cls.tolist()

        # Display result
        if predictions:
            class_name = model.names[int(predictions[0])]
            print(f"Prediction: {class_name}")
        else:
            print("No dark circles detected.")

        # Show image with bounding boxes
        rendered_img = results[0].plot()
        cv2.imshow("Detection Result", rendered_img)

        # Wait until any key is pressed before continuing
        cv2.waitKey(0)
        cv2.destroyWindow("Detection Result")

    elif key == ord('q'):
        print("Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

