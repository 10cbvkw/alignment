import numpy
from scipy.interpolate import RegularGridInterpolator

def generalTransform(image, x_center, y_center, z_center, transform_matrix, method='linear'):

    '''
    image: 3D voxel, (z, x, y)
    '''

    # inverse matrix
    trans_mat_inv = numpy.linalg.inv(transform_matrix)
    # create coordinate meshgrid
    Nz, Ny, Nx = image.shape
    x = numpy.linspace(0, Nx - 1, Nx)
    y = numpy.linspace(0, Ny - 1, Ny)
    z = numpy.linspace(0, Nz - 1, Nz)
    zz, yy, xx = numpy.meshgrid(z, y, x, indexing='ij')
    # calculate transformed coordinate
    coor = numpy.array([xx - x_center, yy - y_center, zz - z_center])
    coor_prime = numpy.tensordot(trans_mat_inv, coor, axes=((1), (0)))
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center
    # get valid coordinates (cell with coordinates within Nx-1, Ny-1, Nz-1)
    x_valid1 = xx_prime>=0
    x_valid2 = xx_prime<=Nx-1
    y_valid1 = yy_prime>=0
    y_valid2 = yy_prime<=Ny-1
    z_valid1 = zz_prime>=0
    z_valid2 = zz_prime<=Nz-1
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
    return_ = numpy.where(valid_voxel > 0)
    print(valid_voxel.shape)
    print(return_)
    print(return_[0].shape)
    print(return_[1].shape)
    print(return_[2].shape)
    print(return_[3].shape)
    assert(0==1)
    z_valid_idx, y_valid_idx, x_valid_idx = numpy.where(valid_voxel > 0) # FIXME too many values to unpack (expected 3)
    # interpolate using scipy RegularGridInterpolator
    image_transformed = numpy.zeros((Nz, Ny, Nx))
    data_w_coor = RegularGridInterpolator((z, y, x), image, method=method)
    interp_points = numpy.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                 yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                 xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
    interp_result = data_w_coor(interp_points)
    image_transformed[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result
    return image_transformed

def rodriguesRotate(image, x_center, y_center, z_center, axis, theta):
    v_length = numpy.linalg.norm(axis)
    if v_length==0:
        raise ValueError("length of rotation axis cannot be zero.")
    if theta==0.0:
        return image
    v = numpy.array(axis) / v_length
    # rodrigues rotation matrix
    W = numpy.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rot3d_mat = numpy.identity(3) + W * numpy.sin(theta) + numpy.dot(W, W) * (1.0 - numpy.cos(theta))
    # transform with given matrix
    return generalTransform(image, x_center, y_center, z_center, rot3d_mat, method='linear')