#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:57:56 2021

@author: tungakin
"""

from __future__ import print_function
import requests
import json
import cv2
import time

service = 'http://127.0.0.1:5000/pass_image'

img = cv2.imread('sample_image.png')
_, img_encoded = cv2.imencode('.png', img)

response = requests.post(service, data=img_encoded.tobytes(), allow_redirects=True,headers={
"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
})
print(json.loads(response.text))