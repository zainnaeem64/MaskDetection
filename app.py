from flask import Flask, request, render_template_string
import os
from models.yolo_utils import load_model, predict_image, draw_bbox_on_image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model()

HTML = '''
<!doctype html>
<title>Mask Detection</title>
<h1>Upload an image for mask detection</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if result_img %}
  <h2>Result:</h2>
  <img src="{{ result_img }}" style="max-width:500px;">
  <h3>Detected classes: {{ detected_classes }}</h3>
  {% if bounding_boxes %}
    <h4>Bounding boxes: {{ bounding_boxes }}</h4>
    <h4>Confidence scores: {{ confidence_scores }}</h4>
  {% endif %}
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result_img = None
    detected_classes = None
    bounding_boxes = None
    confidence_scores = None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            detected_classes, bounding_boxes, confidence_scores, class_ids, results = predict_image(model, img_path, conf=0.25)
            
            # Draw bounding boxes on image and save
            try:
                output_path = draw_bbox_on_image(img_path, bounding_boxes, detected_classes, confidence_scores)
                result_img = output_path
            except Exception as e:
                print(f"Error drawing bounding boxes: {e}")
                # Fallback to original YOLO result
                result_img_path = os.path.join(UPLOAD_FOLDER, 'result_' + file.filename)
                results[0].save(filename=result_img_path)
                result_img = result_img_path
    return render_template_string(HTML, result_img=result_img, detected_classes=detected_classes, 
                                bounding_boxes=bounding_boxes, confidence_scores=confidence_scores)

if __name__ == '__main__':
    app.run(debug=True) 