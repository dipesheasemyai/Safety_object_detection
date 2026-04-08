from ultralytics import YOLO

model = YOLO('/models/yolov8n.pt')

result = model.train(data="configs/yolov8_object_detection.yaml", epochs=100, imgsz=640)
print(result)                                                                                                          