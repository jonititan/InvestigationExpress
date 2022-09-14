import numpy as np
import pandas as pd
import glob
import os

import numpy as np
import cv2 as cv

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
arucoParams = cv.aruco.DetectorParameters_create()
markerlength=0.1

#   display_image(image_with_boxes)

folders = []

for folder in folders:
    images_start_time = float(os.path.split(os.path.split(folder)[0])[-1]) #first image name is time since folder time.
    imagepaths = glob.glob(os.path.join(folder,'*.jpeg')) 
    imagedict={}
    camera_id = os.path.split(image)[-1]
    
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
    metaList = []
    print(folder)
    five_percent = int(len(imagepaths)/20)
    status_5=1
    for index, key in enumerate(sortedImages.keys()):
        if index >= status_5*five_percent:
            print("{}% Processed".format(status_5*5))
            status_5 += 1
        
        img = cv.imread(os.path.join(folder,key))
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        corners, ids, rejected = cv.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
        if len(corners) >= 1:
            print('Found ARUCO marker ID:{} in {}'.format(ids,os.path.join(folder,key)))
            # rvecs,tvecs = cv.aruco.estimatePoseSingleMarkers(corners,markerlength)
            metadata =[index,os.path.join(folder,key),corners, ids, rejected]
            metadata.append(timestamps[index])
            metaList.append(metadata)

    resnet_data = pd.DataFrame(metaList, columns=['index','path','corners','ids', 'rejected','sec'])

    resnet_data.to_csv(os.path.join(folder,'arucodata.csv'))

print("Complete")
# Next steps are detecting marker pose
# https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#ga896ca24f0c1b4b277b6e59d5fe001dd5

