from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')
model.export(format='openvino', imgsz=(384, 640), half=False, dynamic=False, nms=False, batch=1)
print("Export complete. Copy the 'yolo11n_ncnn_model' directory to your Raspberry Pi.")
