#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tungakin
"""
import base64
import BTS
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import jsonpickle
from flask import Flask, render_template, request, redirect, send_file, url_for, Response

app = Flask(__name__)
objectDetectionmodel = torch.hub.load("ultralytics/yolov5", "yolov5s")
depthModel = BTS.BtsController()
depthModel.load_model("models/bts_latest")
depthModel.eval()

@app.route("/pass4depth", methods = ["POST"])
def depthExtract():
    focal = request.form.get('focal')
    img = base64.b64decode(request.form.get('image'))
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prediction = depthModel.predict(img, is_channels_first=False, focal = float(focal), normalize=True) 
    answer = float(prediction[int(prediction.shape[0]/2), int(prediction.shape[1]/2)])
    #Ive tried to get the depth of the middle pixels. Indexing could be changed.
    an_pick = jsonpickle.encode(answer)
    return Response(response=an_pick, status=200)

@app.route("/pass4detection", methods = ["POST"])
def objectDetection():
    img = base64.b64decode(request.form.get('image'))
    nparr = np.frombuffer(img, np.uint8)
    img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    results = objectDetectionmodel(img2)
    results.print()
    result = results.pred[0].cpu().numpy()
    quePoi = np.array([int(img2.shape[0]/2), int(img2.shape[1]/2)])
    response = []
    
    for r in result:
        if (r[1]<=quePoi[0] and r[3]>=quePoi[0] and r[0]<=quePoi[1] and r[2]>=quePoi[1]):
            response.append(results.names[int(r[5])])
        else:
            continue
    
    if len(response)>1:
        response = response[0]
    
    an_pick = jsonpickle.encode(response)
    return Response(response=an_pick, status=200)

if __name__ == '__main__':
    app.run()
