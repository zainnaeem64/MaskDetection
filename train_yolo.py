import os
from models.yolo_utils import ensure_data_downloaded, train_yolo

def main():
    ensure_data_downloaded()
    data_yaml = os.path.join('FaceMask.v2i.yolov11', 'data.yaml')
    train_yolo(data_yaml, model_name='yolo11s.pt', epochs=15, imgsz=640, batch=16, name='facemask')

if __name__ == '__main__':
    main() 