from ultralytics import YOLO
import cv2
model=YOLO('yolov8n.pt')
capture=cv2.VideoCapture('gettyimages-2157600274-640_adpp.mp4')
while True:
    ret,frame=capture.read()
    if not ret:
        break
    results=model.track(
        frame,
        persist=True,
        classes=[0]
    )
    annotated_frame=results[0].plot()
    cv2.imshow('Person Detection', annotated_frame)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()