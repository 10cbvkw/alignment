import sys
sys.path.append(r'/Users/yez/Desktop/alignment/')

import pose_search

import numpy as np
import scipy
from skimage.transform import resize

import scipy.io as scio
import matplotlib.pyplot as plt
img_plt = plt.imread('/Users/yez/Desktop/alignment/SliVo/data/fl.png')

path = r'/Users/yez/Desktop/alignment/SliVo/data/fluorescenceCA1.mat'
fluorescenceCA1 = scio.loadmat(path)['fl']
path = r'/Users/yez/Desktop/alignment/SliVo/data/recondata_cut.mat'
recondata_cut = scio.loadmat(path)['recondata_cut']

fluorescenceCA1 = np.array(fluorescenceCA1)
recondata_cut = np.array(recondata_cut)

R_img = img_plt[:, :, 0]
G_img = img_plt[:, :, 1]
B_img = img_plt[:, :, 2]

image = resize(R_img, output_shape=(256, 256))
plt.imshow(image)
plt.show()

bast_rot, bast_trans = pose_search.opt_theta_trans(recondata_cut, R_img)
bast_rot, bast_trans = pose_search.opt_theta_trans(fluorescenceCA1, R_img)