import sys
import time
sys.path.append(r'/home/yez/alignment/')

import pose_search

import numpy as np
import scipy
import cv2

import scipy.io as scio
import matplotlib.pyplot as plt
img_plt = plt.imread('/home/yez/alignment/SliVo/data/fl.png')

path = r'/home/yez/alignment/SliVo/data/fluorescenceCA1.mat'
fluorescenceCA1 = scio.loadmat(path)['fl']
path = r'/home/yez/alignment/SliVo/data/recondata_cut.mat'
recondata_cut = scio.loadmat(path)['recondata_cut']

fluorescenceCA1 = np.array(fluorescenceCA1)
recondata_cut = np.array(recondata_cut)

R_img = img_plt[:, :, 0]
G_img = img_plt[:, :, 1]
B_img = img_plt[:, :, 2]

image1 = cv2.resize(R_img, (256, 256))
image2 = cv2.resize(G_img, (256, 256))
image3 = cv2.resize(B_img, (256, 256))

recondata_cut = (recondata_cut - recondata_cut.min()) / (recondata_cut.max() - recondata_cut.min()) * 255
fluorescenceCA1 = (fluorescenceCA1 - fluorescenceCA1.min()) / (fluorescenceCA1.max() - fluorescenceCA1.min()) * 255

bast_rot11, bast_trans11, best_z11 = pose_search.opt_theta_trans(recondata_cut, image1)
print(bast_rot11, bast_trans11, best_z11)
'''
bast_rot12, bast_trans12, best_z12 = pose_search.opt_theta_trans(fluorescenceCA1, image1)
print(bast_rot12, bast_trans12, best_z12)

bast_rot21, bast_trans21, best_z21 = pose_search.opt_theta_trans(recondata_cut, image2)
print(bast_rot21, bast_trans21, best_z21)
bast_rot22, bast_trans22, best_z22 = pose_search.opt_theta_trans(fluorescenceCA1, image2)
print(bast_rot22, bast_trans22, best_z22)

bast_rot31, bast_trans31, best_z31 = pose_search.opt_theta_trans(recondata_cut, image3)
print(bast_rot31, bast_trans31, best_z31)
bast_rot32, bast_trans32, best_z32 = pose_search.opt_theta_trans(fluorescenceCA1, image3)
print(bast_rot32, bast_trans32, best_z32)

print('------------------result-------------------')
print(bast_rot11, bast_trans11, best_z11)
print(bast_rot12, bast_trans12, best_z12)
print(bast_rot21, bast_trans21, best_z21)
print(bast_rot22, bast_trans22, best_z22)
print(bast_rot31, bast_trans31, best_z31)
print(bast_rot32, bast_trans32, best_z32)
'''
# FIXME scale of volume and image do not match
