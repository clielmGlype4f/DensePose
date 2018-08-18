from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import requests
import os
import sys
import time
import json
import base64
import common
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
from vis_utils import vis_one_image

c2_utils.import_detectron_ops()

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Densepose args
cfg_file = 'configs/DensePose_ResNet50_FPN_s1x-e2e.yaml'
weights_file = 'weights/DensePose_ResNet50_FPN_s1x-e2e.pkl'
output_dir = 'DensePoseData/infer_out/'
image_ext = 'jpg'
im_or_folder = ''
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)
logger = logging.getLogger(__name__)
merge_cfg_from_file(cfg_file)
cfg.NUM_GPUS = 1
cfg.TEST.BBOX_AUG.ENABLED = False
cfg.MODEL.MASK_ON = False
cfg.MODEL.KEYPOINTS_ON = False

# weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weights_file)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()

# Server configs
PORT = 22100
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)
# Pix2Pix server Configs
PUBLIC_IP = '65.19.181.36'
#PUBLIC_IP = '10.64.15.36'
PIX2PIX_PORT = '23100'
PIX2PIX_ROUTE = '/infer'
pix2pixURL = 'http://' + PUBLIC_IP + ':' + PIX2PIX_PORT + PIX2PIX_ROUTE

# Take in base64 string and return PIL image
def stringToImage(base64_string):
  imgdata = base64.b64decode(base64_string)
  return Image.open(io.BytesIO(imgdata))

# Convert PIL Image to an RGB image(technically a numpy array) that's compatible with opencv
def toRGB(image):
  return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

from scipy.misc import imresize

def main(input_img):
  image = stringToImage(input_img[input_img.find(",")+1:])
  img = toRGB(image)
  logger.info('Processing {} -> {}'.format('New Image', 'Output...'))
  timers = defaultdict(Timer)
  t = time.time()
  size = img.shape[:2]
  print(size)
  #img = imresize(img, (320, 240), interp='bilinear')
  with c2_utils.NamedCudaScope(0):
    cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
      model, img, None, timers=timers
    )
  for key, timer in timers.items():
    print(key, timer.total_time)
  t2 = time.time()
  densepose_img = vis_one_image(img, 'testImage', output_dir, cls_boxes, cls_segms, cls_keyps, cls_bodys, dataset=dummy_coco_dataset, box_alpha=0.3, show_class=True, thresh=0.7, kp_thresh=2)
  t3 = time.time()
  r = requests.post(pix2pixURL, data = {'data': densepose_img})
  logger.info('Inference time: {:.3f}s'.format(t2 - t))
  logger.info('Visualization time: {:.3f}s'.format(t3 - t2))
  logger.info('Pix2pix time: {:.3f}s'.format(time.time() - t3))
  return r

# --- 
# Server Routes
# --- 

# Base route, functions a simple testing 
@app.route('/')
def index():
  return jsonify(status="200", message='Densepose is running', infer_route='/infer')

# Test the model with a fix to see if it's working
@app.route('/test')
def query():
  results = main(None)
  return jsonify(status="200", model='Densepose', response=results)

# When a client socket connects
@socketio.on('connect', namespace='/query')
def new_connection():
  print('Client Connect')
  emit('successful_connection', {"data": "connection established"})

# When a client socket disconnects
@socketio.on('disconnect', namespace='/query')
def disconnect():
  print('Client Disconnect')

# When a client sends data. This should call the main() function
@socketio.on('update_request', namespace='/query')
def new_request(request):
  results = main(request["data"])
  emit('update_response', {"results": results.text})

if __name__ == '__main__':
  socketio.run(app, host='0.0.0.0', port=PORT, debug=False)
