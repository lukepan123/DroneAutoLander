from ultralytics import YOLO

model = YOLO('ugv_yolo11n_V0.pt')  # your existing UGV checkpoint

results = model.train(
    data="dataset2/data.yaml",
    epochs=150,
    imgsz=640,
    batch=8,

    lr0=0.001,
    lrf=0.01,

    patience=30,

    freeze=None,

    degrees=30,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,

    mosaic=0.5,

    plots=True,
    save=True,
)