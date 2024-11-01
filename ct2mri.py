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

    # IMPORTNAT CALC 3 REMINDERS: 
    # NORMAL VECTOR IS PERPENDICULAR TO A PLANE OR A SURFACE, and its used to define orientation of plane
    # orthogonal vector check if two VECTORS are perpendicular to each other: if a dot product b = 0 they are orthogonal
    # conver to normal unit vector using numpys built in functions
    normal = normal / np.linalg.norm(normal)
    
    # if normal is close to the x axis [1,0,0], create an orthogonal vector
    # else then normal is closer to the y axis, create an orthogonal vector with the y axis 
    if not np.allclose(normal, [1, 0, 0]):
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    
    # make a unit vector (normalize it)
    u = u / np.linalg.norm(u)

    # make v which is orthogonal to u and v
    v = np.cross(normal, u)
    
    # Create grid points within the plane
    half_size = plane_size / 2

    #linespace generates sequence of of evenly spaced numbers of range  
    # creates planesize amount of evenly spaced values from -half_size to half_size
    # creates a 1d array of evenly spaced points
    lin_space = np.linspace(-half_size, half_size, int(plane_size))

    # cretea grid in plane using u and v as axes? 
    # .meshgrid produces coordinate matrices from coordinate vectors
    # takes 1d array of evenly spaced points and combines into 2d grid 
    grid_x, grid_y = np.meshgrid(lin_space, lin_space)
    # grid_y = np.meshgrid(lin_space, lin_space)
    
    # Compute 3D coordinates for each point on the 
    #  creates array of shape (N, N, 3), where each row is a 3D point in the plane centered on origin
    # grid x and y determine x and y cords while u and v orient cords in 3d space to essentially create grind
    points = origin + grid_x[..., None] * u + grid_y[..., None] * v

    #  2D array of shape (N, 3) where each row represents the coordinates (in pixel units) of a point in the plane.
    return points.reshape(plane_size * plane_size, 3)
    # return points


def interpolate_image(xyz, data):
    """
    Interpolates the given image data based on the provided coordinates.

    Args:
        xyz (tuple of float): The coordinates in the form of (x, y, z) that need to be converted to integer indices.
        data (numpy.ndarray): The image data to be interpolated.

    Returns:
        numpy.ndarray: The interpolated image data with integer values.
    """
    # create voxel coordinates based of real world cords, every ijk now corresponds to a position in ct_affine
    # A voxel coordinate refers to the position of a voxel (3D pixel) in the index space of a 3D image array, typically in integer units that specify its location along each axis in the array.
    #https://nipy.org/nibabel/reference/nibabel.affines.html#nibabel.affines.apply_affine
    ijk = nib.affines.apply_affine(np.linalg.inv(ct_affine), xyz)
    # round everything to integers for floating point number, indexing array doesnt take floating point
    ijk = np.round(ijk).astype(int)
        
    # Sample data at the nearest indices
    # ijk[:, 0] grabs first column, ijk[:, 1] grabs second column, ijk[:, 2] grabs third column, 
    # first col is the x, 2nd is y
    # effectively create three arrays of cords which is x,y,z of all points we want to sample
    interpolated_data = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    # print("Transformed ijk indices:", ijk[:10])
    
    # Reshape to a 2D grid so we can use in imshow()
    N = int(np.sqrt(len(xyz)))
    # trying to flip the y axis
    # slice_data = interpolated_data.reshape(N, N)
    # slice_data = np.flipud(slice_data)  # Flip along the y-axis
    # return slice_data
    return interpolated_data.reshape(N,N)


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
# print("Sample xyz coordinates:", xyz[:10])


# Interpolate the image data at the grid points
slice_data = interpolate_image(xyz, ct_data)

# Plot the interpolated image. For a given origin and normal, the image should look like a slice with the same origin and normal in Paraview.
plt.figure()
# origin = 'lower' flips the y axis
# plt.imshow(slice_data, cmap='grey', origin = 'lower', extent=[-half_size, half_size, 0, size[0]])

num_ticks = 5
ticks = np.linspace(0, size[0] - 1, num_ticks)  # Pixel positions for tick marks
half_size = (size[0] / 2)  # Physical half-size in your units (e.g., mm)
tick_labels_x = np.linspace(-half_size, half_size, num_ticks)  # X-axis labels from -half_size to +half_size
tick_labels_y = np.linspace(0, size[0], num_ticks)  # Y-axis labels from 0 to plane_size

# Apply tick labels to x and y axes
plt.xticks(ticks=ticks, labels=np.round(tick_labels_x, 2))
plt.yticks(ticks=ticks, labels=np.round(tick_labels_y, 2))

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')


plt.show()





