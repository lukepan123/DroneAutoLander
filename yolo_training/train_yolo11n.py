from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data="dataset1_IRL/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    cache=True,
    seed=0,

    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    patience=50,

    freeze=None,        # try freeze=10 as a second run and compare

    degrees=0,          # was 50
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.5,
    close_mosaic=10,

    plots=True,
)