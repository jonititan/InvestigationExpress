import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os

# modified code from this article
# https://towardsdatascience.com/object-detection-with-tensorflow-model-and-opencv-d839f3e42849

# 'path to input image/video'
folder=r''
IMAGES=glob.glob(os.path.join(folder,'**','**.mp4'),recursive=True) 
print(IMAGES)

# 'path to yolo config file' 
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
CONFIG='yolo/yolov3.cfg'

# 'path to text file containing class names'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
CLASSES='yolo/yolov3.txt'

# 'path to yolo pre-trained weights' 
# wget https://pjreddie.com/media/files/yolov3.weights
WEIGHTS='yolo/yolov3.weights'

for IMAGE in IMAGES:
    outputpath = os.path.split(IMAGE)[0]
    outputfile = os.path.split(IMAGE)[1][:-4]
    print(os.path.exists(CLASSES))
    print(os.path.exists(CONFIG))
    print(os.path.exists(WEIGHTS))
    print(os.path.exists(IMAGE))
    print("Optimised:{}".format(cv.useOptimized()))
    print("Number of Threads:{}".format(cv.getNumThreads()))

    # read class names from text file
    classes = None
    with open(CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
            
    scale = 0.00392
    conf_threshold = 0.5
    nms_threshold = 0.4

    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # function to get the output layer names 
    # in the architecture
    def get_output_layers(net): 
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def processImage(image,index):

        Width = image.shape[1]
        Height = image.shape[0]

        # read pre-trained model and config file
        net = cv.dnn.readNet(WEIGHTS, CONFIG)

        # create input blob 
        blob = cv.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        # set input blob for the network
        net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                
        # apply non-max suppression
        # indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        # for i in indices:
        #     i = i[0]
        #     box = boxes[i]
        #     x = box[0]
        #     y = box[1]
        #     w = box[2]
        #     h = box[3]
        
            # draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        match_data = [class_ids, confidences, boxes,[classes[a] for a in class_ids]]

        return image, match_data

    # open the video file
    cap = cv.VideoCapture(IMAGE)
    # fps = cap.get(cv.CAP_PROP_FPS)
    no_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    ten_percent = int(no_of_frames/10)
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # out = cv.VideoWriter(os.path.join(outputpath,outputfile+'_yolo.mp4'),fourcc, fps, (frame_width,frame_height))
    metaList = [None] * no_of_frames
    index = 0
    status_10 = 1
    while(cap.isOpened()):
        if index >= status_10*ten_percent:
            print("{}% Processed".format(status_10*10))
            status_10 += 1
        ret, frame = cap.read()
        if not ret:
            break
        current_time_msec = cap.get(cv.CAP_PROP_POS_MSEC)
        processedImage,metadata = processImage(frame,index)
        metadata.append(current_time_msec)
        metaList[index] = metadata
        index = index + 1
        # out.write(processedImage)

    # release resources
    cap.release()
    # out.release()
    cv.destroyAllWindows()

    yolo_data = pd.DataFrame(metaList, columns=['class_ids','confidences', 'boxes', 'class_names', 'msec'])

    yolo_data.to_csv(os.path.join(outputpath,outputfile+'_yolodata.csv'))
print("Complete")

