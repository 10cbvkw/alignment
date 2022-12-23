import sys
sys.path.append(r'/home/yez/alignment/')
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import pose_search
from scipy import misc

import numpy as np
import scipy
import cv2
from PIL import Image

import scipy.io as scio
import matplotlib.pyplot as plt
img_plt = plt.imread('/home/yez/alignment/SliVo/data/fl.png')

path = r'/home/yez/alignment/SliVo/data/fluorescenceCA1.mat'
fluorescenceCA1 = scio.loadmat(path)['fl']
path = r'/home/yez/alignment/SliVo/data/recondata_cut.mat'
recondata_cut = scio.loadmat(path)['recondata_cut']

fluorescenceCA1 = np.array(fluorescenceCA1)
recondata_cut = np.array(recondata_cut)
'''
for i in range(256):
    image = fluorescenceCA1[:,:,i]
    max = fluorescenceCA1.max()
    min = fluorescenceCA1.min()
    fl = (image - min)/(max - min) * 255
    fl = fl.astype('uint8')
    im = Image.fromarray(fl)
    im.save("FL/slice" + str(i) + ".jpeg")
'''
'''
for i in range(256):
    image = recondata_cut[:,:,i]
    max = recondata_cut.max()
    min = recondata_cut.min()
    cut = (image - min)/(max - min) * 255
    cut = cut.astype('uint8')
    im = Image.fromarray(cut)
    im.save("CUT/slice" + str(i) + ".jpeg")
'''    
R_img = img_plt[:, :, 0]
G_img = img_plt[:, :, 1]
B_img = img_plt[:, :, 2]

image = cv2.resize(R_img, (256, 256))
GT = (image - image.min())/(image.max()-image.min()) * 255
GT = GT.astype('uint8')
im = Image.fromarray(GT)
im.save("groundtruth.jpeg")

ROTATE_CUT = np.array([[[-0.54937814, -0.20643698, 0.80967119],
    [-0.82922578, 0.01548051, -0.55869935],
    [0.10280208, -0.97833743, -0.17968752]]])
TRANSLATE_CUT = np.array([[29.5267948, -25.1696385]])
Z_CUT = 131

ROTATE_FL = np.array([[[0.27050758,0.92531046,0.26575592],
    [0.59115356,0.05822592,-0.80445461],
    [-0.75984414,0.37471363,-0.53125002]]])
TRANSLATE_FL = np.array([[-2.51785702,-3.08928559]])
Z_FL = 109

def translate_image(image, shifts, L): # checked
    
    """
    image: NY x NX numpy.array
    shift: N * 2 numpy.array
    Return: NY x NX translated at resolution L numpy.array
    """

    N = shifts.shape[0]
    D = image.shape[0]
    images = np.zeros((N, D, D))
    iter = 0
    for shift in shifts:
        new_image = ndimage.shift(input = image, shift = shift)

        mask = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                r = (L-1)*(L-1)
                pixel = ((D-1)/2-i) * ((D-1)/2-i) + ((D-1)/2-j) * ((D-1)/2-j)
                if r < pixel:
                    mask[i,j] = 1

        mask = np.array(mask , dtype = bool)
        img = new_image * mask
        new_image = new_image - img
        images[iter] = new_image
        iter += 1

    return images

IMG_FL = translate_image(image, TRANSLATE_FL, 128)
IMG_CUT = translate_image(image, TRANSLATE_CUT, 128)
FL = (IMG_FL[0] - IMG_FL[0].min()) / (IMG_FL[0].max() - IMG_FL[0].min()) * 255
FL = FL.astype('uint8')
CUT = (IMG_CUT[0] - IMG_CUT[0].min()) / (IMG_CUT[0].max() - IMG_CUT[0].min()) * 255
CUT = CUT.astype('uint8')

im = Image.fromarray(FL)
im.save("fl_trans.jpeg")
im = Image.fromarray(CUT)
im.save("cut_trans.jpeg")

x = np.linspace(-1, 1, 256, endpoint=True)
y = np.linspace(-1, 1, 256, endpoint=True)
z = np.linspace(-1, 1, 256, endpoint=True)
x0, x1, x2 = np.meshgrid(
np.linspace(-1, 1, 256, endpoint=True),
np.linspace(-1, 1, 256, endpoint=True),
np.linspace(-1, 1, 256, endpoint=True),)
coords = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], 1).astype(np.float32)

interp1 = RegularGridInterpolator((x, y, z), fluorescenceCA1, method = 'linear')
interp2 = RegularGridInterpolator((x, y, z), recondata_cut, method = 'linear')

rotated_coords = coords @ ROTATE_FL
rotated_volume_FL = np.zeros((ROTATE_FL.shape[0],256,256,256))
for i in range(ROTATE_FL.shape[0]):
    x_1 = rotated_coords[i,:,0] >= -1
    x_2 = rotated_coords[i,:,0] <= 1
    y_1 = rotated_coords[i,:,1] >= -1
    y_2 = rotated_coords[i,:,1] <= 1
    z_1 = rotated_coords[i,:,2] >= -1
    z_2 = rotated_coords[i,:,2] <= 1
    valid = x_1*x_2*y_1*y_2*z_1*z_2
    rci = rotated_coords[i]
    interp_res = interp1(rci[valid])
    volume_res = np.zeros((256,256,256)).reshape(-1)
    volume_res[valid] = interp_res
    rotated_volume_FL[i] = volume_res.reshape(256,256,256)

for i in range(256):
    image = rotated_volume_FL[0,:,:,i]
    max = rotated_volume_FL.max()
    min = rotated_volume_FL.min()
    cut = (image - min)/(max-min) * 255
    cut = cut.astype('uint8')
    im = Image.fromarray(cut)
    im.save("rotated_fl/slice" + str(i) + ".jpeg")

rotated_coords = coords @ ROTATE_CUT
rotated_volume_CUT = np.zeros((ROTATE_CUT.shape[0],256,256,256))
for i in range(ROTATE_CUT.shape[0]):
    x_1 = rotated_coords[i,:,0] >= -1
    x_2 = rotated_coords[i,:,0] <= 1
    y_1 = rotated_coords[i,:,1] >= -1
    y_2 = rotated_coords[i,:,1] <= 1
    z_1 = rotated_coords[i,:,2] >= -1
    z_2 = rotated_coords[i,:,2] <= 1
    valid = x_1*x_2*y_1*y_2*z_1*z_2
    rci = rotated_coords[i]
    interp_res = interp2(rci[valid])
    volume_res = np.zeros((256,256,256)).reshape(-1)
    volume_res[valid] = interp_res
    rotated_volume_CUT[i] = volume_res.reshape(256,256,256)

for i in range(256):
    image = rotated_volume_CUT[0,:,:,i]
    max = rotated_volume_CUT.max()
    min = rotated_volume_CUT.min()
    cut = (image - min)/(max-min) * 255
    cut = cut.astype('uint8')
    im = Image.fromarray(cut)
    im.save("rotated_cut/slice" + str(i) + ".jpeg")

FL = rotated_volume_FL[0,:,:,Z_FL]
CUT = rotated_volume_CUT[0,:,:,Z_CUT]

FL = (FL - fluorescenceCA1.min()) / (fluorescenceCA1.max() - fluorescenceCA1.min()) * 255
FL = FL.astype('uint8')
CUT = (CUT - recondata_cut.min()) / (recondata_cut.max() - recondata_cut.min()) * 255
CUT = CUT.astype('uint8')

im = Image.fromarray(FL)
im.save("fl.jpeg")
im = Image.fromarray(CUT)
im.save("cut.jpeg")