import numpy as np
import math
from sklearn.mixture import GaussianMixture

def scale_colors(features, new_range=(-1, 1), ch1=(0,255), ch2=(0,255), ch3=(0,255)):
    """ Scale colors to -1 and 1
    """
    features = features.astype(np.float32)
    features[:,0] = (((features[:,0] - ch1[0]) / (ch1[1]-ch1[0])) * (new_range[1]-new_range[0])) + new_range[0]
    features[:,1] = (((features[:,1] - ch2[0]) / (ch2[1]-ch2[0])) * (new_range[1]-new_range[0])) + new_range[0]
    features[:,2] = (((features[:,2] - ch3[0]) / (ch3[1]-ch3[0])) * (new_range[1]-new_range[0])) + new_range[0]
    return features


def process_superpixel(superpixel):
    """ Process superpixel for training data
    """
    return scale_colors(superpixel)

def calculate_score(clf_o, clf_b, superpixel):
    score_o = clf_o.score(np.array(superpixel))
    score_b = clf_b.score(np.array(superpixel))
    if (score_o + score_b) == 0:
        return 0
    else:
        return (score_o / (score_o + score_b))

def compute_color_consistency(img1, sp_map1, msid1,
                              img2, sp_map2, msid2,
                              lambda_param=1.0,
                              n_components=15,
                              random_state=42):
    """ Compute the color consistency between two superpixels.
    """
    features1 = img1[sp_map1==msid1]
    features2 = img2[sp_map2==msid2]
    
    features1 = scale_colors(features1, new_range=(0, 1))
    features2 = scale_colors(features2, new_range=(0, 1))
    
    # Using two GMM objects
    gmm1 = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm1.fit(features1)
    u1 = gmm1.means_
    
    # Sort the columns independently
    u1idx = u1.argsort(axis=0)
    u1 = u1[u1idx, np.arange(u1idx.shape[1])]
    
    gmm2 = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm2.fit(features2)
    u2 = gmm2.means_

    # Sort the columns independently
    u2idx = u2.argsort(axis=0)
    u2 = u2[u2idx, np.arange(u2idx.shape[1])]
    
    # l2-norm squared
    u = (u1 - u2)
    u_norm_sq = np.sum(u**2)

    c = math.exp(-lambda_param * u_norm_sq)
    return c