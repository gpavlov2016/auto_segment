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

def generate_W(vid, spm, num_frames=1, frame_skip=1):
    maps = spm.maps
    W = np.zeros((spm.total_sp, spm.total_sp))
    prev_image = None
    log("Working with",num_frames,"frames and frame skip",frame_skip)
    for i in get_frames_range(vid, num_frames=num_frames):
        frame = i * frame_skip
        if prev_image is None:
            prev_image = get_smaller_image(vid, frame)
        else:
            map_id1 = i - 1
            sp_map1 = maps[map_id1]
            map_id2 = map_id1 + 1
            log("Map",map_id1,"and",map_id2)

            image = get_smaller_image(vid, frame)

            point_pairs = ep.find_point_matches(prev_image, image, mtx,
                                                n_matches=100, n_features=1000)
            F, mask = ep.find_fundamental_matrix(point_pairs)

            # Find all superpixel ids and centers in the first image.
            # While doing so, set weights to neighboring superpixels.
            sp_locs = []
            sids = []
            start = time.time()
            total_sp = np.max(sp_map1) + 1
            log("Setting W with one image (",total_sp,"superpixels)...")
            counter = 0
            for msid in range(total_sp):
                counter+=1
                sid = spm.msid2sid(map_id1, msid)
                
                sp_loc, _ = sp.find_superpixel_center(sp_map1, msid)
                sp_locs.append([sp_loc])
                sids.append(sid)

                # Neighbor map superpixel ids
                nmsids = sp.find_neighbors(sp_map1, msid)
                log("  Computing color consistency of", len(nmsids),
                    "superpixel neighbors...")
                substart = time.time()
                for nmsid in nmsids:
                    nsid = spm.msid2sid(map_id1, nmsid)
                    W[sid, nsid] = \
                      mo.compute_color_consistency(
                        prev_image, sp_map1, msid,
                        prev_image, sp_map1, nmsid
                      )
                subdiff = (time.time() - substart)
                log("    Took", subdiff,
                    "seconds (",counter,"/",total_sp,")")
            diff = (time.time() - start)
            log("Completed setting W of superpixel neighbors in", diff, "seconds.")
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

def create_graph(W):
    g = maxflow.Graph[float](W.shape[0], np.count_nonzero(W))
    nodes = g.add_nodes(W.shape[0])

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] > 0:
                g.add_edge(nodes[i], nodes[j], W[i, j], W[j, i])
    return g, nodes