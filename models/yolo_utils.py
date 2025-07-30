import os
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

def ensure_data_downloaded():
    data_dir = 'FaceMask.v2i.yolov11'
    data_yaml = os.path.join(data_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        from roboflow import Roboflow
        load_dotenv()
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError('API_KEY not found in .env file')
        rf = Roboflow(api_key=api_key)
        project = rf.project('facemask-e8tat')
        version = project.version(1)
        dataset = version.download('yolov11')
        print(f"Dataset downloaded to {dataset.location}")
    else:
        print(f"Dataset already exists at {data_dir}")

def train_yolo(data_yaml, model_name='yolo11s.pt', epochs=15, imgsz=640, batch=16, name='facemask'):
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, name=name)

def get_best_weights():
    """Return the path to the best YOLO weights file."""
    weights_path = os.path.join('Notebook', 'runs', 'detect', 'facemask3', 'weights', 'best.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join('Notebook', 'runs', 'detect', 'facemask', 'weights', 'best.pt')
    return weights_path

def load_model(weights_path=None):
    """Load a YOLO model from weights."""
    if weights_path is None:
        weights_path = get_best_weights()
    return YOLO(weights_path)

def predict_image(model, image_path, conf=0.25):
    """Run prediction on a single image and return detected class names, bounding boxes, and confidence scores."""
    results = model.predict(source=image_path, conf=conf)
    detected_classes = []
    bounding_boxes = []
    confidence_scores = []
    class_ids = []
    
    for result in results:
        if result.boxes is not None and result.boxes.cls is not None and len(result.boxes.cls) > 0:
            # Extract class names
            classes = [model.names[int(cls)] for cls in result.boxes.cls]
            detected_classes.extend(classes)
            
            # Extract bounding boxes (x1, y1, x2, y2 format)
            boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy array
            bounding_boxes.extend(boxes.tolist())
            
            # Extract confidence scores
            conf_scores = result.boxes.conf.cpu().numpy()
            confidence_scores.extend(conf_scores.tolist())
            
            # Extract class IDs
            cls_ids = result.boxes.cls.cpu().numpy()
            class_ids.extend(cls_ids.tolist())
    
    if not detected_classes:
        detected_classes = ['No objects detected']
        bounding_boxes = []
        confidence_scores = []
        class_ids = []
    
    return detected_classes, bounding_boxes, confidence_scores, class_ids, results

def draw_bbox_on_image(image_path, bounding_boxes, detected_classes, confidence_scores, output_path=None):
    """Draw bounding boxes with class labels and confidence scores on the image."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Define colors for different classes (BGR format)
    colors = {
        'mask': (0, 255, 0),      # Green
        'no_mask': (0, 0, 255),   # Red
        'incorrect_mask': (0, 255, 255)  # Yellow
    }
    
    # Draw bounding boxes
    for i, (bbox, class_name, conf) in enumerate(zip(bounding_boxes, detected_classes, confidence_scores)):
        if class_name == 'No objects detected':
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this class
        color = colors.get(class_name, (255, 255, 255))  # White as default
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label with class name and confidence
        label = f"{class_name}: {conf:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the image
    if output_path is None:
        # Create output path in same directory as input
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_bbox.jpg")
    
    cv2.imwrite(output_path, image)
    return output_path 