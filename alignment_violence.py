import sys
sys.path.append(r'/Users/yez/Desktop/alignment/')
import rotate3D_scipy
import shift_grid
import so3_grid


import numpy as np
import scipy

import scipy.io as scio
import matplotlib.pyplot as plt
img_plt = plt.imread('/Users/yez/Desktop/alignment/SliVo/data/fl.png')

path = r'/Users/yez/Desktop/alignment/SliVo/data/fluorescenceCA1.mat'
fluorescenceCA1 = scio.loadmat(path)['fl']
path = r'/Users/yez/Desktop/alignment/SliVo/data/recondata_cut.mat'
recondata_cut = scio.loadmat(path)['recondata_cut']

fluorescenceCA1 = np.array(fluorescenceCA1)
recondata_cut = np.array(recondata_cut)

volume_fl = fluorescenceCA1
volume_re = recondata_cut



err = np.zeros(256, images.shape[0], rot.shape[0])
volume_re = volume_re.transpose(2, 0, 1)
volume_re_rotated = rotate3D_scipy.generalTransform(volume_re, 128, 128, 128, rot, method='linear')
for z_dim in range(256):
    y_hat = volume_re_rotated[z_dim, :, :] 
    err[z_dim] = -(images * y_hat).sum(-1) / y_hat.std(-1)








    



