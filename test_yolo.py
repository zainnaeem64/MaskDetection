import os
import sys
from models.yolo_utils import load_model, predict_image, draw_bbox_on_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_yolo.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)
    model = load_model()
    detected_classes, bounding_boxes, confidence_scores, class_ids, results = predict_image(model, image_path, conf=0.5)
    print("Detected classes:", detected_classes)
    if bounding_boxes:
        print("Bounding boxes:", bounding_boxes)
        print("Confidence scores:", confidence_scores)
        print("Class IDs:", class_ids)
        
        # Draw bounding boxes on image and save
        try:
            output_path = draw_bbox_on_image(image_path, bounding_boxes, detected_classes, confidence_scores)
            print(f"Image with bounding boxes saved to: {output_path}")
        except Exception as e:
            print(f"Error drawing bounding boxes: {e}")

if __name__ == '__main__':
    main() 