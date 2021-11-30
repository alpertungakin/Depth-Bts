#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tungakin
"""

from __future__ import print_function
from urllib3._collections import HTTPHeaderDict
import requests
import json
import cv2
import time
import base64

def optimizeFocal(sourceIm, destSize, focal):
    ratio0 = destSize[1]/sourceIm.shape[0]
    ratio1 = destSize[0]/sourceIm.shape[1]
    foc0 = focal*ratio0
    foc1 = focal*ratio1
    foc = (foc0 + foc1)/2
    return foc

depthApi = 'http://127.0.0.1:5000/pass4depth'
objcApi = 'http://127.0.0.1:5000/pass4detection'
img = cv2.imread('2.jpg')
img1 = cv2.resize(img,(352, 264))
_, img_encoded_depth = cv2.imencode('.jpg', img1)
_, img_encoded_objc = cv2.imencode('.jpg', img1)
focal = optimizeFocal(img, (352, 264), 3460)
dataObjct = HTTPHeaderDict()
dataObjct.add('image', base64.b64encode(img_encoded_objc))
dataDepth = HTTPHeaderDict()
dataDepth.add('focal', focal)
dataDepth.add('image', base64.b64encode(img_encoded_depth))
responseDepth = requests.post(depthApi, data=dataDepth, allow_redirects=True,headers={
"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
})
responseObjc = requests.post(objcApi, data=dataObjct, allow_redirects=True,headers={
"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
})

print(json.loads(responseDepth.text))
print(responseDepth.elapsed.total_seconds())
print(json.loads(responseObjc.text))
print(responseObjc.elapsed.total_seconds())
print(responseDepth.elapsed.total_seconds()+responseObjc.elapsed.total_seconds())
