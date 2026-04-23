import ultralytics

model = ultralytics.YOLO("yolov8m.pt")

results = model.train(
    data="C:\\Users\\yurin\\Downloads\\teste\\dataset\\parking.yaml",
    epochs=50,
    imgsz=640
)

print("Treino concluído!")