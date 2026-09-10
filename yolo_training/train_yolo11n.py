from ultralytics import YOLO

model = YOLO('ugv_yolo11n.pt')  # your existing UGV checkpoint

results = model.train(
    data='dataset1/data.yaml',
    epochs=50,        # fewer than a from-scratch run — it's already trained, not starting cold
    imgsz=640,
    batch=16,
    lr0=0.001,        # lower than the default (0.01) so updates are gentler on existing weights
    patience=15,      # early stopping if val performance plateaus, to avoid overfitting the new subset
    freeze=10         # freezes the first 10 layers (the backbone) so general features stay intact;
                       # only the detection head adapts to your new data
)