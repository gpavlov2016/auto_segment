import superpixels as sp
import cv2
import imageio
import numpy as np
import pdb
from sklearn.mixture import GaussianMixture
from helpers import *
import epipolar as ep
import pickle
import models as mo
import time

def run():
    filename = './data/toothpick.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    # Number of frames to use. NONE for all frames.
    num_frames = 1

    # How many frames to skip e.g. if the value is 5,
    # then we take frame 0, 5, 10, etc.
    frame_skip = 10

    # List of superpixel maps
    maps = []

    # List of Edge Adjacency Matrix
    W = []

    # List of ID of superpixel that overlaps with the fixation point
    obj_ids = []

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
    
    # Learn object and background color models
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
    W = np.zeros((spm.total_sp, spm.total_sp))
    prev_image = None
    log("Working with",num_frames,"frames")
    for i in get_frames_range(vid, num_frames=num_frames):
        frame = i * frame_skip

        sp_map= maps[i]
        image = get_smaller_image(vid, frame)

        # Find all superpixel ids and centers in the first image.
        # While doing so, set weights to neighboring superpixels.
        sp_locs = []
        sids = []
        start = time.time()
        total_sp = np.max(sp_map) + 1
        log("Setting W with one image (",total_sp,"superpixels)...")
        counter = 0
        for msid in range(total_sp):
            counter+=1
            sid = spm.msid2sid(i, msid)
            
            sp_loc, _ = sp.find_superpixel_center(sp_map, msid)
            sp_locs.append([sp_loc])
            sids.append(sid)

            # Neighbor map superpixel ids
            nmsids = sp.find_neighbors(sp_map, msid)
            log("  Computing color consistency of", len(nmsids),
                "superpixel neighbors...")
            substart = time.time()
            for nmsid in nmsids:
                nsid = spm.msid2sid(i, nmsid)
                W[sid, nsid] = \
                  mo.compute_color_consistency(
                    image, sp_map, msid,
                    image, sp_map, nmsid
                  )
            subdiff = (time.time() - substart)
            log("    Took", subdiff,
                "seconds (",counter,"/",total_sp,")")
        diff = (time.time() - start)
        log("Completed setting W of superpixel neighbors in", diff, "seconds.")

        if prev_image is None:
            prev_image = get_smaller_image(vid, frame)
        else:
            map_id1 = i - 1
            sp_map1 = maps[map_id1]
            map_id2 = map_id1 + 1
            log("Map",map_id1,"and",map_id2)

            point_pairs = ep.find_point_matches(prev_image, image, mtx,
                                                n_matches=100, n_features=1000)
            F, mask = ep.find_fundamental_matrix(point_pairs)

            log("Computing epilines for", len(sp_locs), "superpixels...")
            start = time.time()
            # Compute epilines given superpixel centers.
            lines = ep.compute_epilines(sp_locs, F)
            diff = (time.time() - start)
            log("Epilines computed.")

            # For each line + superpixel of the first image,
            # get second image's superpixels that overlap with
            # that line. We then update Edge Adjacency Matrix W
            # with color consistency between each image 1 and 2 superpixel
            # pair.
            start = time.time()
            log("Setting W with image pair (",len(sids),"superpixels)...")
            counter = 0
            for line, img1_sid in zip(lines, sids):
                counter+=1
                map_id1, msid1 = spm.sid2msid(img1_sid)
                
                if len(maps) > map_id2:
                    sp_map2 = maps[map_id2]
                    msid_matches = \
                      ep.get_superpixels_on_epiline(sp_map2, line)
                    log("  Computing color consistency of", len(msid_matches),
                         "superpixel matches...")
                    substart = time.time()
                    for msid2 in msid_matches:
                        img2_sid = spm.msid2sid(map_id2, msid2)
                        W[img1_sid, img2_sid] = \
                          mo.compute_color_consistency(
                            prev_image, sp_map1, msid1,
                            image, sp_map2, msid2
                          )
                    subdiff = (time.time() - substart)
                    log("    Took", subdiff,
                        "seconds (",counter,"/",len(lines),")")
            diff = (time.time() - start)
            log("Completed setting W of image pair's superpixels in", diff, "seconds.")
            log("")
            prev_image = image

    np.savetxt("./data/W.csv", W, delimiter=',')
    with open("./data/maps.pkl", "wb") as out:
        pickle.dump(maps, out)
    
    # END - Create W
    # ================

if __name__ == '__main__':
    run()