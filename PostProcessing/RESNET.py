import numpy as np
import pandas as pd
import glob
import os
# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub


# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
strategy = tf.distribute.MirroredStrategy()


module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(module_handle).signatures['default']
def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img
# https://www.tensorflow.org/hub/tutorials/object_detection



def run_detector(detector, path, strat):
    img = load_img(path)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    #   start_time = time.time()
    with strat.scope():
        result = detector(converted_img)
    #   end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    #   print("Found %d objects." % len(result["detection_scores"]))
    #   print("Inference time: ", end_time-start_time)

    return result["detection_class_entities"],result["detection_scores"], result["detection_boxes"]
#   image_with_boxes = draw_boxes(
#       img.numpy(), result["detection_boxes"],
#       result["detection_class_entities"], result["detection_scores"])


#   display_image(image_with_boxes)

folders = []

for folder in folders:
    images_start_time = float(os.path.split(os.path.split(folder)[0])[-1]) #first image name is time since folder time.
    imagepaths = glob.glob(os.path.join(folder,'*.jpeg')) 
    imagedict={}
    
    for image in imagepaths:
        filename = os.path.split(image)[-1]
        try:
            imagedict[filename]=float(filename[:-5])
        except ValueError as ve:
            print(ve)
            print(filename)
            print(filename[-5])
            raise
        
    sortedImages = dict(sorted(imagedict.items(), key=lambda x:x[1]))
    timestamps=list(sortedImages.values())
    metaList = [None] * len(imagepaths)
    print(folder)
    five_percent = int(len(imagepaths)/20)
    status_5=1
    for index, key in enumerate(sortedImages.keys()):
        if index >= status_5*five_percent:
            print("{}% Processed".format(status_5*5))
            status_5 += 1
        metadata = list(run_detector(detector, os.path.join(folder,key), strategy))
        metadata.append(timestamps[index])
        metaList[index] = metadata

    resnet_data = pd.DataFrame(metaList, columns=['class_ids','confidences', 'boxes', 'sec'])

    resnet_data.to_csv(os.path.join(folder,'resnetdata.csv'))

print("Complete")

