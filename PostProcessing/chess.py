import numpy as np
import pandas as pd
import glob
import os
from scipy.spatial import ConvexHull, distance
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

print("Optimised:{}".format(cv.useOptimized()))
print("Number of Threads:{}".format(cv.getNumThreads()))
def order_points(pts):
    
    # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :] # modified
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[-2:, :] # modified [2:, :] was original but caused failure if more than 4 coordinates supplied
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def process_checkerboard(image,cornerPoints, colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))):
                
            corners2 = cv.cornerSubPix(image,cornerPoints,(11,11),(-1,-1),criteria)
            #pairwise distance
            center = corners2.mean(axis=0)
            points = np.array([a[0] for a in corners2])
            hull = ConvexHull(points)
            rect_corners = points[hull.vertices]
            rect_points = order_points(rect_corners)
            justchess = four_point_transform(image,rect_corners)
            fullframe = image.copy()
            for ((x, y), color) in zip(rect_points, colors):
                cv.circle(fullframe, (int(x), int(y)), 5, color, -1)

                    
            return justchess, corners2, fullframe, center


def four_point_transform(image, pts):
    # https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    rad_array = np.full(xy.shape[0],radians)

    x, y = xy.T
    offset_x, offset_y = np.full(xy.shape,origin).T
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(rad_array)
    sin_rad = np.sin(rad_array)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboardrows = 6
chessboardcols = 9
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objl = np.zeros((chessboardrows*chessboardcols,3), np.float32)
objl[:,:2] = np.mgrid[0:chessboardcols,0:chessboardrows].T.reshape(-1,2)
objp = np.zeros((chessboardcols*chessboardrows,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardrows,0:chessboardcols].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


#   display_image(image_with_boxes)
cameradict = {'rgb_idnumber':['right_idnumber','left_idnumber'],} # i had several cameras and I had their ids in here
folders = [] # add your list of folders in here


camera_data = []
for folder in folders:
    images_start_time = float(os.path.split(os.path.split(folder)[0])[-1]) #first image name is time since folder time.
    imagepaths = glob.glob(os.path.join(folder,'*.jpeg')) 
    imagedict={}
    root_path, camera_id = os.path.split(folder)
    if camera_id in cameradict.keys():
        rgb=True
    else:
        rgb=False
    foundChess=False
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
        
        
        if rgb:
            img = cv.imread(os.path.join(folder,key))
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        else:
            gray = cv.imread(os.path.join(folder,key),0) # read as greyscale
        #landscape orientation chessboard
        retl, cornersl= cv.findChessboardCorners(gray, (chessboardcols,chessboardrows),None)
        #portrait orientation chessboard
        retp, cornersp= cv.findChessboardCorners(gray, (chessboardrows,chessboardcols),None)

        
        # If found, add object points, image points (after refining them)
        if retp or retl: 
            if not foundChess:
                foundChess=True
                try:
                    newpath = os.path.join(folder,'chess')
                    os.mkdir(newpath)
                except FileExistsError as FEE:
                    print('Folder {} already created'.format(newpath))

        if retp:
            justchesspic, cornersdata, frame, center = process_checkerboard(gray,cornersp)
            no_intersections = len(cornersdata)
            orientation = 'portrait'
            # if rgb:
            #     frame = cv.drawChessboardCorners(gray,  (chessboardrows,chessboardcols),  cornersdata, retp) 
            cv.imwrite(os.path.join(folder,'chess','portrait_{}_'.format(no_intersections)+key),justchesspic)
            cv.imwrite(os.path.join(folder,'chess','portrait_full_{}_'.format(no_intersections)+key),frame)
            # cornersdata = rotate_around_point_highperf(cornersdata, np.pi/2, center) # this creates floats which calibratecamera can't handle
            objpoints.append(objp)
            imgpoints.append(cornersdata)
            metadata =[index,os.path.join(folder,key),orientation,center,cornersdata,timestamps[index],retp,no_intersections]
            metaList.append(metadata)
            retp = False
            

        if retl:
            justchesspic, cornersdata, frame, center = process_checkerboard(gray,cornersl)
            no_intersections = len(cornersdata)
            orientation = 'landscape'
            # if rgb:
            #     frame = cv.drawChessboardCorners(gray,  (chessboardrows,chessboardcols),  cornersdata, retp) 
            cv.imwrite(os.path.join(folder,'chess','landscape_{}_'.format(no_intersections)+key),justchesspic)
            cv.imwrite(os.path.join(folder,'chess','landscape_full_{}_'.format(no_intersections)+key),frame)
            objpoints.append(objl)
            imgpoints.append(cornersdata)
            metadata =[index,os.path.join(folder,key),orientation,center,cornersdata,timestamps[index],retl,no_intersections]
            metaList.append(metadata)
            retl = False
            
            
       
           

    corner_data = pd.DataFrame(metaList, columns=['index','path','chessboardfound','center','corners', 'sec','cornerstatus','no_intersections'])

    corner_data.to_pickle(os.path.join(folder,'chess_data.pickle'))
    if len(objpoints) >=1:
        assert len(objpoints) == len(imgpoints)
        try:
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        except:
            print(imgpoints[0])

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        totalError = mean_error/len(objpoints)
        camera_data.append({'session start':images_start_time,'camera':camera_id,'mtx':mtx,'dist':dist,'error':totalError})
        print("Total error: {}".format(totalError) )
        print("Camera Matrix: \n{}".format(mtx))
        print("Distortion Coefficents: \n{}".format(dist))
        print('Rotation Vector:\n{}'.format(rvecs))
        print("Translation Vector:\n{}".format(tvecs))
        with open(os.path.join(folder,'calib_data.txt'),'w') as outputfile:
                outputfile.write("Camera ID: {}".format(os.path.split(folder)[1]))
                outputfile.write("Camera Matrix: \n{}\n".format(mtx))
                outputfile.write("Distortion Coefficents: \n{}\n".format(dist))
                outputfile.write("Total error: \n{}\n".format(totalError))
                outputfile.write('Rotation Vector:\n{}'.format(rvecs))
                outputfile.write("Translation Vector:\n{}".format(tvecs))

cameraCalibrations = pd.DataFrame.from_records(camera_data)
print(cameraCalibrations.head())
cameraCalibrations.to_pickle('allCameraCalibrationData.pickle')
print("Complete")
