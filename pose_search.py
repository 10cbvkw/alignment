import numpy as np
import os
from scipy import ndimage

from scipy.spatial.transform import Rotation as R
import sys
import torch

sys.path.append(r'/home/yez/alignment/')
from scipy.interpolate import RegularGridInterpolator
import shift_grid
import so3_grid

def rot_2d(angle, outD, device=None): # checked
    rot = np.zeros((outD, outD))
    rot[0, 0] = np.cos(angle)
    rot[0, 1] = -np.sin(angle)
    rot[1, 0] = np.sin(angle)
    rot[1, 1] = np.cos(angle)
    return rot

Lmin = 64 # FIXME input parameters
Lmax = 128 # FIXME input parameters
D = 256 # FIXME input parameters
tilt = None
base_healpy = 1
t_extent = 32
t_ngrid = 7
niter = 10
nkeptposes = 24
loss_fn = "msf"
t_xshift = 0
t_yshift = 0
FAST_INPLANE = True

so3_base_quat = so3_grid.grid_SO3(base_healpy)
base_quat = (so3_grid.s2_grid_SO3(base_healpy) if FAST_INPLANE else so3_base_quat)
r = R.from_quat(so3_base_quat)
so3_base_rot = r.as_matrix()
r = R.from_quat(base_quat)
base_r = r.as_matrix()
nbase = len(base_quat)
base_inplane = so3_grid.grid_s1(base_healpy)
base_s = shift_grid.base_shift_grid(base_healpy - 1, t_extent, t_ngrid, xshift=t_xshift, yshift=t_yshift)
t_extent = t_extent
t_ngrid = t_ngrid

Lmin = Lmin
Lmax = Lmax
niter = niter
tilt = tilt
nkeptposes = nkeptposes
loss_fn = loss_fn
_so3_neighbor_cache = {}  # for memoization
_shift_neighbor_cache = {}  # for memoization
os.environ['CUDA_VISIBLE_DEVICE']='2'

def eval_grid(volume, images, rot, NQ, L):

    """
    volume: 3D colume
    images: T x Npix
    rot: Q x 3 x 3 rotation matrics 
    NQ: number of slices evaluated for each image
    L: radius of fourier components to evaluate
    """

    nz = 1

    def compute_err(volume, images, rot, nz):

        """
        images: T x Npix
        rot: Q x 3 x 3 rotation matrics 
        """
        
        err = np.zeros((256, rot.shape[0], images.shape[0]))
        x = np.linspace(-1, 1, 256, endpoint=True)
        y = np.linspace(-1, 1, 256, endpoint=True)
        z = np.linspace(-1, 1, 256, endpoint=True)
        interp = RegularGridInterpolator((x, y, z), volume, method = 'linear')
        # FIXME use cuda interp = RegularGridInterpolator((x, y, z), volume_, method = 'linear')
        x0, x1, x2 = np.meshgrid(
            np.linspace(-1, 1, 256, endpoint=True),
            np.linspace(-1, 1, 256, endpoint=True),
            np.linspace(-1, 1, 256, endpoint=True),)
        coords = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], 1).astype(np.float32)
        rotated_coords = coords @ rot
        rotated_coords = rotated_coords # to satisfy grid_sample
        rotated_volume = np.zeros((rot.shape[0],256,256,256))

        err = torch.from_numpy(err).cpu()
        err = err.cuda()
        volume_ = torch.from_numpy(volume)
        volume_ = volume_.cuda()
        volume_ = volume_.view(1,1,volume_.shape[0],volume_.shape[1],volume_.shape[2])
        images = torch.from_numpy(images)
        images = images.cuda()
        for i in range(rotated_coords.shape[0]):
            print(i)
            grid = rotated_coords[i].reshape(1,1,1,rotated_coords.shape[1],rotated_coords.shape[2])
            grid = torch.from_numpy(grid)
            grid = grid.cuda()
            interp_res = torch.nn.functional.grid_sample(volume_, grid, mode='bilinear', align_corners=True)
            rotated_volume = interp_res.view(256,256,256)
            rotated_volume = rotated_volume.permute(2,0,1)

            for ind_ in range(images.shape[0]):
                err[:, i, ind_] = -(images[ind_] * rotated_volume).mean(axis = [1,2], keepdim = False)

        err = (err.cpu()).numpy()    
        err_z = []
        for z_dim in range(256):
            err_z.append(err[z_dim].min())
        value = min(err_z)
        keepz = err_z.index(value)
        print('keepz = ', keepz)
        return err[keepz], keepz

    err, keepz = compute_err(volume ,images, rot, nz)
    return err, keepz  # nzxTxQ

def mask_image(image, L): # checked

    """
    image: NY x NX numpy.array
    Return: NY x NX masked at resolution L numpy.array
    """
    
    D = image.shape[0]
    mask = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            r = (L-1)*(L-1)
            pixel = ((D-1)/2-i) * ((D-1)/2-i) + ((D-1)/2-j) * ((D-1)/2-j)
            if r < pixel:
                mask[i,j] = 1
    mask = np.array(mask , dtype = bool)
    img = image * mask
    image = image - img
    return image
    
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

def rotate_image(image, angle, L): # checked

    """
    image: NY x NX numpy.array
    angle: float 
    Return: NY x NX rotated at resolution L numpy.array
    """
    
    D = image.shape[0]
    new_image = ndimage.rotate(input = image, angle = angle)

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
    
    return new_image

def get_neighbor_so3(quat, s2i, s1i, res): # checked

    """Memoization of so3_grid.get_neighbor."""
    
    key = (int(s2i), int(s1i), int(res))
    if key not in _so3_neighbor_cache:
        _so3_neighbor_cache[key] = so3_grid.get_neighbor(quat, s2i, s1i, res)
    return _so3_neighbor_cache[key]

def get_neighbor_shift(x, y, res): # checked
    
    """Memoization of shift_grid.get_neighbor."""
    
    key = (int(x), int(y), int(res))
    if key not in _shift_neighbor_cache:
        _shift_neighbor_cache[key] = shift_grid.get_neighbor(x, y, res - 1, t_extent, t_ngrid)
    return _shift_neighbor_cache[key]

def subdivide(quat, q_ind, cur_res):

    """
    Subdivides poses for next resolution level

    Inputs:
        quat (N x 4 np.array): quaternions
        q_ind (N x 2 np.array): index of current S2xS1 grid
        cur_res (int): Current resolution level

    Returns:
        quat  (N x 8 x 4) np.array
        q_ind (N x 8 x 2) np.array
        rot   (N * 8 x 3 x 3) np.array
        trans (N * 4 x 2) np.array
    """
    
    N = quat.shape[0]

    assert len(quat.shape) == 2 and quat.shape == (N, 4), quat.shape
    assert len(q_ind.shape) == 2 and q_ind.shape == (N, 2), q_ind.shape

    # get neighboring SO3 elements at next resolution level
    neighbors = [
        get_neighbor_so3(quat[i], q_ind[i][0], q_ind[i][1], cur_res)
        for i in range(len(quat))
    ]
    quat = np.array([x[0] for x in neighbors])  # 8x4
    q_ind = np.array([x[1] for x in neighbors])  # 8x2
    r = R.from_quat(quat.reshape(-1, 4))
    rot = r.as_matrix()

    assert len(quat.shape) == 3 and quat.shape == (N, 8, 4), quat.shape
    assert len(q_ind.shape) == 3 and q_ind.shape == (N, 8, 2), q_ind.shape
    assert len(rot.shape) == 3 and rot.shape == (N * 8, 3, 3), rot.shape

    return quat, q_ind, rot
    
def keep_matrix(loss, max_poses): # checked
    
    """
    Inputs:
        loss (T, Q): numpy.array of losses for each translation and rotation.

    Returns:
        keep (2, max_poses): bool numpy.array of rotations to keep, along with the best translation for each
    """
    
    shape = loss.shape
    assert len(shape) == 2
    best_loss = loss.min(axis = 1) 
    best_trans_idx = loss.argmin(axis = 1)
    flat_loss = best_loss.reshape(-1)

    flat_idx = flat_loss.argsort()[0:max_poses]
    flat_idx = flat_idx.reshape(-1)

    keep_idx = np.zeros((len(shape), max_poses))
    keep_idx[1] = flat_idx
    keep_idx[0] = best_trans_idx[flat_idx]
    keep_idx = keep_idx.astype(int)
    print('best_loss = ', best_loss)
    print('best_trans_idx = ', best_trans_idx)
    print('flat_idx = ', flat_idx)
    print('keep_idx = ', keep_idx)
    return keep_idx

def getL(iter_): # checked
    L = Lmin + int(iter_ / niter * (Lmax - Lmin))
    return min(L, D // 2)

def opt_theta_trans(volume_, image):
    
    base_rot = base_r
    base_shifts = base_s
    nkeptposes = 8
    
    # Compute the loss for all poses
    L = getL(0)
    loss, keepz = eval_grid(
        volume=volume_,
        images=translate_image(image, base_shifts, L), 
        rot=base_rot, 
        NQ=nbase, 
        L=L, 
        )
    
    keepT, keepQ = keep_matrix(loss, nkeptposes)
    quat = so3_base_quat[keepQ]
    q_ind = so3_grid.get_base_ind(keepQ, base_healpy) 
    trans = base_shifts[keepT]
    shifts = base_shifts

    for iter_ in range(1, niter + 1):
        print(iter_)
        L = getL(iter_)
        quat, q_ind, rot = subdivide(quat, q_ind, iter_ + base_healpy - 1)
        quat = quat.reshape(-1,4)
        q_ind = q_ind.reshape(-1,2)
        shifts /= 2
        trans_ = np.zeros((trans.shape[0] * shifts.shape[0], 2))
        for i in range(trans.shape[0]):
            for j in range(shifts.shape[0]):
                trans_[i * nkeptposes + j, :] = trans[i, :] + shifts[j, :]
        trans = trans_
        rot = rot
        loss, keepz = eval_grid(
            volume=volume_,
            images=translate_image(image, trans, L),  
            rot=rot,
            NQ=8,
            L=L)

        nkeptposes = nkeptposes if iter_ < niter else 1
        keepT, keepQ = keep_matrix(loss, nkeptposes)
        quat = quat[keepQ]
        q_ind = q_ind[keepQ]
        trans = trans[keepT]

    bestT, bestQ = keep_matrix(loss, 1)
    if niter == 0:
        best_rot = so3_base_rot[bestQ]
        best_trans = base_shifts[bestT]
    else:
        best_rot = rot[bestQ].reshape(3, 3)
        best_trans = trans
    best_z = keepz
    return best_rot, best_trans, best_z
