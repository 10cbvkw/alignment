import numpy as np
from scipy import ndimage

from scipy.spatial.transform import Rotation as R
import sys

sys.path.append(r'/Users/yez/desktop/alignment/')
import shift_grid
import rotate3D_scipy
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
t_extent = 5
t_ngrid = 7
niter = 5
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

def eval_grid(volume, images, rot, NQ, L):

    """
    volume: 3D colume
    images: T x Npix
    rot: Q x 3 x 3 rotation matrics 
    NQ: number of slices evaluated for each image
    L: radius of fourier components to evaluate
    """

    # FIXME mask
    nz = 1

    def compute_err(volume, images, rot, nz):

        """
        images: T x Npix
        rot: Q x 3 x 3 rotation matrics 
        """

        err = np.zeros((256, images.shape[0], rot.shape[0]))
        volume = volume.transpose(2, 0, 1)
        volume_rotated = rotate3D_scipy.generalTransform(volume, 128, 128, 128, rot, method='linear')
        for z_dim in range(256):
            y_hat = volume_rotated[z_dim, :, :] 
            err[z_dim] = -(images * y_hat).sum(-1) / y_hat.std(-1)
        
        err_z = []
        for z_dim in range(256):
            err_z.append(err[z_dim].sum())

        value = max(err_z)
        keepz = err_z.index(value)
        
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

    return keep_idx

def getL(iter_): # checked
    L = Lmin + int(iter_ / niter * (Lmax - Lmin))
    return min(L, D // 2)

def opt_theta_trans(volume_, image):
    
    base_rot = base_r
    base_shifts = base_s
    
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
        shifts /= 2
        trans = trans + shifts
        rot = rot
        loss, keepz = eval_grid(
            volume=volume_,
            images=translate_image(image, trans, L),  
            rot=rot,
            NQ=8,
            L=L)

        # FIXME loss(256, T, Q)选出256个中小的一个负相关的值

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
        best_rot = rot.reshape(8, 3, 3)[bestQ]
        best_trans = trans
    best_z = keepz

    return best_rot, best_trans, best_z
