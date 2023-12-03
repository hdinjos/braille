from flask import Flask, request, Response
import time
import PIL
import torch
from ultralyticsplus import YOLO, render_result
import json
import numpy as np
import torch
import os
import json

app = Flask(__name__, static_folder="assets")

def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    # copy values from troublesome "boxes" object to numpy array
    new_boxes = np.zeros(boxes.shape)
    new_boxes[:, :4] = boxes.xywh.numpy()  # first 4 channels are xywh
    new_boxes[:, 4] = boxes.conf.numpy()  # 5th channel is confidence
    new_boxes[:, 5] = boxes.cls.numpy()  # 6th channel is class which is last channel

    # sort according to y coordinate
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]

    # find threshold index to break the line
    y_threshold = np.mean(new_boxes[:, 3]) // 2
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]

    # cluster according to threshold_index
    boxes_clustered = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    for cluster in boxes_clustered:
        # sort according to x coordinate
        cluster = cluster[cluster[:, 0].argsort()]
        boxes_return.append(cluster)

    return boxes_return

def load_model(model_path):
    """load model from path"""
    model = YOLO(model_path)
    return model


def load_image(image_path):
    """load image from path"""
    image = PIL.Image.open(image_path)
    return image


@app.route("/")
def start():
    return {"msg" : "Welcome to our server"}


ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/predict")
def predict():
    try:
        if 'file' not in request.files:
            error = {
                "err" : True,
                "msg" : "field file not found"
            }
            return Response(status=400, mimetype="application/json", response=json.dumps(error))
       
        file = request.files['file']
        if file.filename == '':
            error = {
                "err" : True,
                "msg" : "image file required"
            }
            return Response(status=400, mimetype="application/json", response=json.dumps(error))
            
        if not allowed_file(file.filename):
            error = {
                "err" : True,
                "msg" : "file type not allowed"
            }
            return Response(status=400, mimetype="application/json", response=json.dumps(error))

        basedir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(basedir, 'assets/yolov8m.pt')
        model = load_model(model_path)
        model.overrides["conf"] = 0.10  # NMS confidence threshold
        model.overrides["iou"] = 0.20  # NMS IoU threshold
        model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        model.overrides["max_det"] = 1000  # maximum number of detections per image

        image = load_image(file)

        with torch.no_grad():
            res = model.predict(image, save=False, save_txt=False, exist_ok=True, conf=0.10)
            boxes = res[0].boxes
            list_boxes = parse_xywh_and_class(boxes)
            str_predict = []
            for box_line in list_boxes:
                box_classes = box_line[:, -1]
                for each_class in box_classes:
                    str_predict.append(model.names[int(each_class)])
            
        return str_predict

    except Exception as ex:
        return Response(status=400, mimetype="application/json", response=json.dumps(ex))
