from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

results = model.train(
    data=r".\datasets\pokemoncarddetector.v7i.yolov11\data.yaml",
    epochs=100,          
    imgsz=640,          
    device="cpu",        
    patience=10,
    # workers=4,         
)

model.val()

onnx_path = model.export(format="onnx", imgsz=640) 

infer_model = YOLO(onnx_path)

for i in range(1, 6):
    results = infer_model.predict(
        source=fr'.\src\test{i}.jpg',
        imgsz=640,
        conf=0.25
    )
    results[0].show()

print(infer_model.names)