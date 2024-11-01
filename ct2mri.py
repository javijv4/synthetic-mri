#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/10/30 16:22:52

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib


def grid_in_plane(origin, normal, plane_size):
    """
    Generates a grid of points in a plane defined by an origin and a normal vector.
    Note that the grid is centered at the origin and the points are spaced the same in both in-plane directions.
    You want to make sure the grid is large enough to cover the entire image (plane_size argument).

    Args:
        origin (numpy.ndarray): A 3-element numpy array representing the coordinates of the origin point of the plane.
        normal (numpy.ndarray): A 3-element numpy array representing the normal vector of the plane.
        plane_size (float): The size of the plane in pixel units

    Returns:
        numpy.ndarray: A 2D array of shape (N, 3) where each row represents the coordinates (in pixel units) of a point in the plane.
    """
    normal = normal / np.linalg.norm(normal)
    
    # Create two orthogonal vectors in the plane using the normal vector
    if not np.allclose(normal, [1, 0, 0]):
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Create grid points within the plane
    half_size = plane_size / 2
    lin_space = np.linspace(-half_size, half_size, int(plane_size))
    grid_x, grid_y = np.meshgrid(lin_space, lin_space)
    
    # Compute 3D coordinates for each point on the plane
    points = origin + grid_x[..., None] * u + grid_y[..., None] * v
    return points.reshape(-1, 3)


def interpolate_image(xyz, data):
    """
    Interpolates the given image data based on the provided coordinates.

    Args:
        xyz (tuple of float): The coordinates in the form of (x, y, z) that need to be converted to integer indices.
        data (numpy.ndarray): The image data to be interpolated.

    Returns:
        numpy.ndarray: The interpolated image data with integer values.
    """
    ijk = nib.affines.apply_affine(np.linalg.inv(ct_affine), xyz)
    ijk = np.round(ijk).astype(int)
        
    # Sample data at the nearest indices
    interpolated_data = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    print("Transformed ijk indices:", ijk[:10])
    
    # Reshape to a 2D grid (assuming square grid as above)
    return interpolated_data.reshape(int(np.sqrt(len(xyz))), int(np.sqrt(len(xyz))))


# Load CT image
ct_img = nib.load('data/labels.nii.gz')
ct_data = ct_img.get_fdata()
ct_affine = ct_img.affine

size = ct_data.shape

# Define a plane given its normal and a point on it
origin = np.array([0, 0, 0])  
normal = np.array([0, 0, 1])

# Define a grid of points that live in the plane
xyz = grid_in_plane(origin, normal, size[0])
print("Sample xyz coordinates:", xyz[:10])


# Interpolate the image data at the grid points
slice_data = interpolate_image(xyz, ct_data)

# Plot the interpolated image. For a given origin and normal, the image should look like a slice with the same origin and normal in Paraview.
plt.figure()
plt.imshow(slice_data, cmap='grey')
plt.show()





