from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train3/weights/best.pt')
cap = cv2.VideoCapture("video3.mp4")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, conf=0.25)
    cv2.imshow("YOLO", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()