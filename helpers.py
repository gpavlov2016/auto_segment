import cv2

def log(*args):
    print(*args)

def get_smaller_image(vid, i, ratio=0.25):
    image = vid.get_data(i)
    image = cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
    return image

def show_image(image):
    cv2.imshow('out', image)

    cv2.waitKey(0)

def get_frames_range(vid, num_frames=None):
    """ Get list of frame indexes from a video.
    """
    return range(num_frames)


def visual_hull_not_converged():
    """ TODO: Not sure what this is for, so we will just use range here.
              Stops after iterates 2 times.
    """
    value = True
    for i in range(2):
        if i == 2:
            value = False
        else:
            value = True
        yield(value)

def calibrate(glob_path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows = 7
    cols = 7
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    import pylab
    import imageio
    images = []
    import glob

    #for i in range(0, len(images), 1):
    #images = ['IMG_20171211_151001.jpg']
    images = glob.glob(glob_path)
    print(images)
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img', gray)
        #cv2.waitKey(500)

        print('Processing calibration images')
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    return  cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)