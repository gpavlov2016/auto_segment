""" 
"""

import superpixels as sp
import cv2
import imageio
import numpy as np
import pdb
from sklearn.mixture import GaussianMixture
from helpers import *
import graph_helpers as gh
import epipolar as ep
import pickle
import models as mo
import time

def run():
    filename = './data/toothpick.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Number of frames to use. NONE for all frames.
    num_frames = 2

    # How many frames to skip e.g. if the value is 5,
    # then we take frame 0, 5, 10, etc.
    frame_skip = 10

    # List of superpixel maps
    maps = []

    # List of Edge Adjacency Matrix
    W = []

    # List of ID of superpixel that overlaps with the fixation point
    obj_ids = []

    # Number of backgrounds superpixels to be picked randomly in each
    # graph-cut step.
    num_random_bg

    # Get superpixels and fixation points
    # ================
    for i in get_frames_range(vid, num_frames=num_frames):
        frame = i * frame_skip
        image = get_smaller_image(vid, frame)
        sp.num_superpixels = 100
        sp_map = sp.extract_superpixels(image)
        maps.append(sp_map)

        # fixation point is assumed to be at the center of the image.
        fp = sp_map[int(image.shape[0]/2.0), int(image.shape[1]/2.0)]
        obj_ids.append(fp)

    # END - Get superpixels and fixation points
    # ================
    
    # Learn object and background color models from fixation points
    # ================
    clf_o = GaussianMixture(n_components=1)
    clf_b = GaussianMixture(n_components=1)

    for i in get_frames_range(vid, num_frames=num_frames):
        frame = i * frame_skip
        sp_map = maps[i]
        obj_id = obj_ids[i]

        # TODO: Another alternative is to keep the images in a list. Which one would
        #       be better?
        image = get_smaller_image(vid, frame)
        
        # fitting the models
        # ------------------
        # Currently, we fit the models by simply including all objects into
        # `clf_o.fit()` and backgrounds to `clf_b.fit()`.

        objects, backgrounds = sp.split_superpixels(image, sp_map, [obj_id])

        for o in objects:
            clf_o.fit(o)
        for b in backgrounds:
            clf_b.fit(b)

        # END - fitting the models
        # ------------------

    # END - Learn object and background color models
    # ================

    # Create W
    # ================

    # Calibration needs only be done once.
    # ret, mtx, dist, rvecs, tvecs = calibrate('../../calib/*.jpg')
    # print(mtx)
    mtx = np.array([[ 633.14676506,    0.,            407.2286842 ],
                    [   0.,            627.07243572,  281.21216339],
                    [   0.,            0.,            1.        ]])

    spm = sp.SuperpixelsMaps(maps)

    # while visual_hull_not_converged(prev_visual_hull, visual_hull):
    for i in range(2):
        # Evaluate object likelihood and keep them as 
        W = gh.generate_W(vid, clf_o, clf_b, obj_ids,
                          num_frames=num_frames, frame_skip=frame_skip)

        # Perform graph-cut to label superpixels
        # ================
        g, nodes = gh.create_graph(W)
        bg_ids = spm.pick_random_bg(obj_ids, num=num_random_bg)
        for obj_id in obj_ids:
            # For each each node with the same id as the object
            # superpixel, connect it to the source node with weight
            # of 1.0. We then randomly pick one background superpixel,
            # get a node with a similar id, and then run a min-cut operation.
            # This operation will divide the nodes into 0 - closer to object
            # and 1 - closer to background.
            g.add_tedge(nodes[obj_id], 1000, 0)
            for bg_id in bg_ids:
                g.add_tedge(nodes[bg_id], 0, 1000)

        # ================


    # END - Create W
    # ================

if __name__ == '__main__':
    run()