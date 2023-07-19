import io

from PIL import Image
import numpy as np
import uuid
from pathlib import Path
from flask import Flask, request, send_file
import cv2

from model import YOLOv8_face, draw_detections

Path("./processed").mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

DETECTION_URL = '/v1/blur/image'


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return

    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        model = YOLOv8_face("./weights/yolov8n-face.onnx", conf_thres=0.45, iou_thres=0.5)

        # Detect Objects
        src_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        boxes, scores, classids, kpts = model.detect(src_image)

        blurred_image = draw_detections(src_image, boxes, scores, kpts)
        image_id = uuid.uuid4()
        file_path = f'processed/{image_id}.jpeg'

        cv2.imwrite(file_path, blurred_image)

        return send_file(file_path, mimetype='image/jpeg')

    return "<p>Could not process request</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)