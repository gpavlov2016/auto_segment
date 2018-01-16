import cv2
import numpy as np
import random

# Settings
# These settings can be updated when importing this module e.g.
# import superpixels as sp
# sp.num_superpixels = 3000
num_superpixels = 4000
num_levels = 4
prior = 2
num_histogram_bins = 7
num_iterations = 520

def extract_superpixels(image):
    """ Extract superpixels from an image in the form of a list of maps

    The returned maps can then be used to initialize a Superpixels object.
    """
    global num_superpixels, num_levels, prior, num_histogram_bins, num_iterations
    
    height,width,channels = image.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
                                               num_superpixels, num_levels, prior, num_histogram_bins)
    color_img = np.zeros((height,width,3), np.uint8)
    color_img[:] = (0, 0, 255)
    seeds.iterate(image, num_iterations)
    maps = seeds.getLabels()
    return maps


def draw_superpixels(image, labels):
    # labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(False)

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)

    return result

def process_superpixel(superpixel):
    """ Process superpixel for training data
    """
    return superpixel

def split_superpixels(image, maps, obj_ids, bg_sample_ratio = 0.5):
    """ Split the superpixels in an image into object and background  superpixels.

    The number of samples taken from the background 
    """
    max_sp_id = np.max(maps)
    sp_o = []
    sp_b = []
    for i in range(max_sp_id+1):
        superpixel = image[maps == i]
        if i in obj_ids:
            sp_o.append(process_superpixel(superpixel))
        else:
            sp_b.append(process_superpixel(superpixel))
    sp_b = np.array(sp_b)
    random.sample(range(len(sp_b)), int(len(sp_b) * bg_sample_ratio))
    return (sp_o, sp_b)

def find_neighbors(sp_map, sp_id):
    y = sp_map == sp_id  # convert to Boolean

    rolled = np.roll(y, 1, axis=0)          # shift down
    rolled[0, :] = False             
    z = np.logical_or(y, rolled)

    rolled = np.roll(y, -1, axis=0)         # shift up 
    rolled[-1, :] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, 1, axis=1)          # shift right
    rolled[:, 0] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, -1, axis=1)         # shift left
    rolled[:, -1] = False
    z = np.logical_or(z, rolled)

    neighbors = set(np.unique(np.extract(z, sp_map))) - set([sp_id])
    return neighbors

def highlight_superpixel(sp_map, msid):
    """ Set superpixel msid to 200 while setting
        others to 0.
    """
    cp = sp_map.copy()
    cp[np.where(cp!=msid)] = -1
    cp[np.where(cp==msid)] = 200
    cp[np.where(cp==-1)] = 0
    cp = cp.astype(np.uint8)
    thresh = cv2.threshold(cp, 100, 255, cv2.THRESH_BINARY)[1]

    return thresh

def highlight_superpixels(sp_map, msids):
    himg = np.zeros((sp_map.shape[0], sp_map.shape[1]))
    for sid in msids:
        himg1 = highlight_superpixel(sp_map, sid)
        himg = np.sum((himg, himg1), axis=0)
    return himg

def find_superpixel_center(sp_map, msid):
    """ Find the coordinate of superpixel center.
    """
    himg = highlight_superpixel(sp_map, msid)
    cnts = cv2.findContours(himg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    main_cnt = cnts[0]
    for cnt in cnts:
        if len(cnt) > len(main_cnt):
            main_cnt = cnt
    main_cnt = np.array(main_cnt).reshape((-1,1,2)).astype(np.int32)
    M = cv2.moments(main_cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return ((cX, cY), main_cnt)

class SuperpixelsMaps:
    """ A class that translates between a list of maps and superpixels.
    """
    def __init__(self, maps):
        self.maps = maps

        # Total number or superpixels
        self.total_sp = 0

        for sp_map in maps:
            self.total_sp += np.max(sp_map)+1

    def sid2msid(self, sid):
        """ Converts superpixel id to map & local superpixel id
        """
        remaining_sid = sid + 1
        map_id = -1
        msid = 0
        while (remaining_sid > 0 or map_id >= len(self.maps)):
            map_id += 1
            sp_map = self.maps[map_id]
            # print("map_id:", map_id)
            # print(sp_map)
            max_sp_id = np.max(sp_map)
            num_map_sps = max_sp_id + 1
            # print("num sps in this map:", num_map_sps)
            # print("remaining_sid:", remaining_sid)
            msid = min(remaining_sid, num_map_sps) - 1
            # print("found msid:", msid)
            remaining_sid -= num_map_sps
        return (map_id, msid)

    def msid2sid(self, map_id, msid):
        """ Converts map id & local superpixel id to superpixel id.
        """
        # print("map_id:",map_id,"msid:",msid)
        sid = 0
        for i in range(map_id+1):
            sp_map = self.maps[i]
            max_sp_id = np.max(sp_map)
            num_map_sps = max_sp_id + 1
            # print("num_map_sps:", num_map_sps)
            # print("i:",i)
            if i >= (map_id):
                sid += msid
            else:
                sid += num_map_sps
            # print("sid:",sid)
        return sid

    def pick_random_except(self, ids, num=1, random_state=None):
        """ Pick random superpixels except the ones defined in `ids`

        Useful for when we want to randomly get background superpixels.
        """
        all_ids = list(range(self.total_sp))
        all_sel_ids = [sel_id for sel_id in all_ids if sel_id not in ids]

        if num > len(all_sel_ids):
            return all_sel_ids
        else:
            random.seed(random_state)
            return random.sample(all_sel_ids, num)

    def pick_local_random_except(self, map_id, ids, num=1):
        """ Pick random superpixels in a map except the ones defined in `ids`
        """
        all_ids = list(range(self.total_sp))
        all_sel_ids = [sel_id for sel_id in all_ids if sel_id not in ids]
        
        if num > len(all_sel_ids):
            return all_sel_ids
        else:
            return random.sample(all_sel_ids, num)

def test_spm():
    """ Test SuperpixelsMaps object
    """
    # image 1 has 5 superpixels,
    # image 2 has 6,
    # and image 3 has 4.
    maps = [
        [[0, 0, 0, 1],
         [2, 2, 2, 3],
         [4, 2, 3, 3],
         [4, 2, 2, 3]],
        [[0, 0, 0, 0],
         [2, 2, 1, 1],
         [4, 2, 3, 3],
         [4, 5, 2, 3]],
        [[0, 0, 0, 0],
         [2, 1, 1, 0],
         [2, 2, 3, 3],
         [2, 2, 3, 3]],
    ]

    spm = SuperpixelsMaps(maps)
    assert(spm.total_sp == 15)

    map_id, msid = spm.sid2msid(12)
    assert(map_id == 2)
    assert(msid == 1)

    map_id, msid = spm.sid2msid(0)
    print(msid)
    assert(map_id == 0)
    assert(msid == 0)

    sid = spm.msid2sid(2, 2)
    assert(sid == 13)

    sid = spm.msid2sid(0, 1)
    assert(sid == 1)

def test_w():
    map1 = np.array([
        [0, 1, 1],
        [1, 1, 2],
        [1, 1, 2]
    ])
    map2 = np.array([
        [0, 1, 1],
        [2, 1, 1],
        [2, 1, 1]
    ])

    maps = [map1, map2]
    spm = SuperpixelsMaps(maps)
    W = w_from_superpixels(spm)

    W_result = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    print("ideal W:")
    print(W_result)
    print("W:")
    print(W)
    assert(np.sum(W - W_result) == 0)

if __name__ == '__main__':
    from pprint import pprint
    import pdb
    test_spm()
    test_w()