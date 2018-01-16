import numpy as np
import cv2


def find_point_matches(img1, img2, K,
                       n_features=1000, n_matches=10):
    """ Find point matches between two images.

    Args:
        - img1: Image 1
        - img2: Image 2
        - K: mtx output from `cv2.calibrateCamera()` function
    """
    w = img1.shape[1]
    h = img1.shape[0]
    corners = [[0,0], [0,w-1], [h-1,w-1], [h-1, 0]]
    # Initiate ORB detector
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=n_features)
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    # draw only keypoints location,not size and orientation
    vis = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = np.zeros_like(img2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                           matches[:n_matches], img3,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    focal = K[0,0]
    ppx = K[0,2]
    ppy = K[1,2]
    alignedLeft = {'pt': [], 'des': []}
    alignedRight = {'pt': [], 'des': []}
    leftBackReference = []
    rightBackReference = []
    #Arrange matching points in aligned arrays:
    for i, match in enumerate(matches[:n_matches]):
        qid = match.queryIdx
        tid = match.trainIdx
        #print('qid, tid:', qid, tid)
        alignedLeft['pt'].append(kp1[qid].pt)
        alignedLeft['des'].append(des1[qid,:])
        alignedRight['pt'].append(kp2[tid].pt)
        alignedRight['des'].append(des2[tid,:])
    pts1 = np.array(alignedLeft['pt']).reshape(n_matches, 1, -1).astype(np.float32)
    pts2 = np.array(alignedRight['pt']).reshape(n_matches, 1, -1).astype(np.float32)

    return (pts1, pts2)

def find_fundamental_matrix(point_matches):
    """ Find Fundamental Matrix from point matches.
    """
    F, mask = cv2.findFundamentalMat(point_matches[0], point_matches[1],
                                     cv2.FM_RANSAC)
    return (F, mask)

def compute_epilines(pts, F):
    # print(np.array(pts)[0])
    # print(np.array(pts).astype(np.float32))
    pts = np.array(pts).astype(np.float32)
    lines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1, 3)
    return lines

def get_superpixels_on_epiline(sp_map, line): 
    sids = []
    for x in range(sp_map.shape[1]):
        y = -(line[2] + line[0] * x) / line[1]
        y = int(round(y))
        if y >= 0 and y < sp_map.shape[0]:
            sids.append(sp_map[y, x])
    sids = list(set(sids))
    return sids