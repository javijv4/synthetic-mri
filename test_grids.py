#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/12/06 10:37:47

@author: Javiera Jilberto Vallejos 
'''

import os
from matplotlib import pyplot as plt
import numpy as np
import functions as fn
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D

# User inputs
ct_path = 'data/labels.nii.gz'
out_path = 'data/'
labels = {'LV': 1, 'RV': 3, 'Aorta': 6}
inplane_spacing = 10.0       # Use this when generating the grid
out_of_plane_spacing = 8.0
number_of_slices = 13

# Create output directory
if not os.path.exists(out_path):
    os.makedirs(out_path, exists=True)

# Read CT image
ct_data, ct_affine, pixdim = fn.readFromNIFTI('data/labels.nii.gz')
size = ct_data.shape
plane_size = size[0]/3


# Calculate position of LV/RV/Aorta
spatial_info = fn.calculate_spatial_information(ct_data, ct_affine, labels['LV'], labels['RV'], labels["Aorta"])

# Grabbing centroid and normal for all views
sa_normal_origin, la_2ch_normal_origin, la_3ch_normal_origin, la_4ch_normal_origin = fn.get_view_normal_origin(spatial_info)

# Create grid for all views
sa_grid, sa_affine = fn.grid_in_plane(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size)
la_2ch_grid, la_2ch_affine = fn.grid_in_plane(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size)
la_3ch_grid, la_3ch_affine = fn.grid_in_plane(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size)
la_4ch_grid, la_4ch_affine = fn.grid_in_plane(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size)
print(sa_affine)

# Make ij grid
i = np.arange(int(np.sqrt(len(sa_grid))))
I, J = np.meshgrid(i, i)
ij = np.column_stack((I.ravel(), J.ravel()))
ijk = np.column_stack((ij, np.zeros_like(ij[:,0])))

# Transform grid to xyz
sa_grid_xyz = nib.affines.apply_affine(sa_affine, ijk)
la_2ch_grid_xyz = nib.affines.apply_affine(la_2ch_affine, ijk)
la_3ch_grid_xyz = nib.affines.apply_affine(la_3ch_affine, ijk)
la_4ch_grid_xyz = nib.affines.apply_affine(la_4ch_affine, ijk)

print(sa_grid)
fig = fn.show_point_cloud(sa_grid, color='red', label='SA')
fn.show_point_cloud(la_2ch_grid, fig=fig, color='blue', label='2CH')
fn.show_point_cloud(la_3ch_grid, fig=fig, color='green', label='3CH')
fn.show_point_cloud(la_4ch_grid, fig=fig, color='yellow', label='4CH')

fn.show_point_cloud(sa_grid_xyz, fig=fig, color='black', marker_symbol='cross', label='SA affine')
fn.show_point_cloud(la_2ch_grid_xyz, fig=fig, color='black', marker_symbol='cross', label='2CH affine')
fn.show_point_cloud(la_3ch_grid_xyz, fig=fig, color='black', marker_symbol='cross', label='3CH affine')
fn.show_point_cloud(la_4ch_grid_xyz, fig=fig, color='black', marker_symbol='cross', label='4CH affine')
fig.show()