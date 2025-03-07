from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-seg.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='/home/lwf/nanting/yolov8/ultralytics-main/default.yaml', epochs=100,imgsz=512)  # train the model
