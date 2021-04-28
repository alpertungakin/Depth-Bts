#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:49:50 2021

@author: tungakin
"""
import BTS
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import jsonpickle
from flask import Flask, render_template, request, redirect, send_file, url_for, Response

app = Flask(__name__)
model = BTS.BtsController()
model.load_model("models/bts_latest")
model.eval()

@app.route("/pass_image", methods=["POST"])
def passImage():
    r = request
    img = r.data
    nparr = np.frombuffer(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV loads images as BGR by default
    prediction = model.predict(img, is_channels_first=False, normalize=True) # Dont forget to normalize images
    answer = float(prediction[int(prediction.shape[0]/2), int(prediction.shape[1]/2)])
    #Ive tried to get the depth of middle pixel of the image. Index could be changed.
    an_pick = jsonpickle.encode(answer)
    return Response(response=an_pick, status=200)


if __name__ == '__main__':
    app.run()
