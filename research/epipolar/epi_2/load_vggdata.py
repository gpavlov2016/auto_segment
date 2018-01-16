# http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf
# line 129

import numpy as np
import camera
from PIL import Image
# load some images
im1 = np.array(Image.open('../../../data/merton1/images/001.jpg'))
im2 = np.array(Image.open('../../../data/merton1/images/002.jpg'))

# load 2D points for each view to a list
points2D = [np.loadtxt('../../../data/merton1/2D/00'+str(i+1)+'.corners').T for i in range(3)]
# load 3D points
points3D = np.loadtxt('../../../data/merton1/3D/p3d').T
# load correspondences
corr = np.genfromtxt('../../../data/merton1/2D/nview-corners', dtype='int', missing_values='*')
# load cameras to a list of Camera objects
P = [camera.Camera(np.loadtxt('../../../data/merton1/2D/00'+str(i+1)+'.P')) for i in range(3)]