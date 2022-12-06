import numpy as np
import scipy
import sys

sys.path.append(r'/Users/yez/desktop/alignment/')
import shift_grid
import shift_grid3
# import lie_tools
import so3_grid

np.set_printoptions(threshold=np.inf)

L = 5
image = np.array(range(900)).reshape(30,-1)

Lmin = 64 # FIXME input parameters
Lmax = 128 # FIXME input parameters
D = 256 # FIXME input parameters
tilt = None
base_healpy = 1
t_extent = 5
t_ngrid = 7
niter = 5
nkeptposes = 24
loss_fn = "msf"
t_xshift = 0
t_yshift = 0
FAST_INPLANE = True

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
        new_image = scipy.ndimage.shift(input = image, shift = shift)

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

base_shifts = shift_grid.base_shift_grid(base_healpy - 1, t_extent, t_ngrid, xshift=t_xshift, yshift=t_yshift)
images=translate_image(image, base_shifts, L)
print(images)