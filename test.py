import numpy as np
import scipy
import sys
import torch 

sys.path.append(r'/home/yez/alignment/')
import shift_grid
import shift_grid3
# import lie_tools
import so3_grid

input = torch.tensor(range(720)).double()
input = input.view(6, 10, 12)
mul = torch.tensor(range(120)).double()
mul = mul.view(10, 12)
mul1 = input[3]
output = mul * input
print(output[0].sum())
print(output[1].sum())
print(output[2].sum())
print(output[3].sum())
print(output[4].sum())
print(output[5].sum())
print((mul*mul1).sum())
'''
input = torch.tensor(range(720)).double()
input = input.view(1, 1, 6, 10, 12)
# input = input.numpy()
grid = torch.tensor([-0.99, -0.99, -0.99]).double()
grid = grid.view(1, 1, 1, 1, 3)
interp = torch.nn.functional.grid_sample(input, grid, mode='bilinear', align_corners=True)
print(interp)
grid = torch.tensor([-0.99, -0.99, -0.99, 0.99, 0.99, 0.99]).double()
grid = grid.view(1, 1, 2, 1, 3)
interp = torch.nn.functional.grid_sample(input, grid, mode='bilinear', align_corners=True)
print(interp)
'''

'''
a = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,0.]])
b = torch.tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8],[0.9,0.0]])

print(a.shape, b.shape)
print(a.unsqueeze(1)+b.unsqueeze(0))
'''
'''
x0, x1, x2 = np.meshgrid(
            np.linspace(-1, 1, 256, endpoint=True),
            np.linspace(-1, 1, 256, endpoint=True),
            np.linspace(-1, 1, 256, endpoint=True),
        )
coords = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], 1).astype(np.float32)
print(coords)
'''


'''
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
'''

# CPU version
'''
        for i in range(rot.shape[0]):
            print(i)
            x_1 = rotated_coords[i,:,0] >= -1
            x_2 = rotated_coords[i,:,0] <= 1
            y_1 = rotated_coords[i,:,1] >= -1
            y_2 = rotated_coords[i,:,1] <= 1
            z_1 = rotated_coords[i,:,2] >= -1
            z_2 = rotated_coords[i,:,2] <= 1
            valid = x_1*x_2*y_1*y_2*z_1*z_2
            rci = rotated_coords[i]
            interp_res = interp(rci[valid])
            volume_res = np.zeros((256,256,256)).reshape(-1)
            volume_res[valid] = interp_res
            rotated_volume[i] = volume_res.reshape(256,256,256)
            for z_dim in range(256):
                y_hat = rotated_volume[i, :, :, z_dim]
                for ind_ in  range(images.shape[0]):
                    err[z_dim, i, ind_] = -(images[ind_] * y_hat).sum() 
'''

