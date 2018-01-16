import cv2
import numpy as np

def get_superpixels_in_row(sp_map, y):
    sids = []
    for x in range(sp_map.shape[1]):
        sids.append(sp_map[y, x])
    sids = sorted(list(set(sids)))
    return sids

def highlight_superpixel(sp_map, msid, value=None, thresh=True):
    """ Set superpixel msid to a value while setting
        others to 0.
    """
    cp = sp_map.copy()
    cp[np.where(cp!=msid)] = -1
    if value is None:
        value = 255
    cp[np.where(cp==msid)] = value
    cp[np.where(cp==-1)] = 0
    cp = cp.astype(np.uint8)
    if thresh:
        cp = cv2.threshold(cp, 100, 255, cv2.THRESH_BINARY)[1]

    return cp

def highlight_superpixels_hsl(sp_map, msids, values=None, max_value=1.0,
                              max_intensity=255, min_intensity=50, h=90, s=100):
    """
    """
    h_channel = np.zeros((sp_map.shape[0], sp_map.shape[1]))
    s_channel = np.zeros((sp_map.shape[0], sp_map.shape[1]))
    l_channel = np.zeros((sp_map.shape[0], sp_map.shape[1]))
    for i, sid in enumerate(msids):
        if values is None:
            value = None
        else:
            value = min_intensity + (max_intensity - min_intensity) * (values[i]/max_value)
        l_highlight = highlight_superpixel(sp_map, sid, value=value, thresh=False)
        l_channel = np.sum((l_channel, l_highlight), axis=0)

        h_highlight = highlight_superpixel(sp_map, sid, value=h, thresh=False)
        h_channel = np.sum((h_channel, h_highlight), axis=0)

        s_highlight = highlight_superpixel(sp_map, sid, value=s, thresh=False)
        s_channel = np.sum((s_channel, s_highlight), axis=0)

    hls_img = np.stack((h_channel, l_channel, s_channel), axis=2).astype(np.uint8)
    rgb_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)
    return rgb_img
