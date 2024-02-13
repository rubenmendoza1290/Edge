from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.export(format="onnx")

if __name__ == "__main__":
    main()
