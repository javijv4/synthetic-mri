#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/10/30 16:22:52
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
inplane_spacing = 3.0       # Use this when generating the grid
spacing = 2.0       # Use this when generating the grid
out_of_plane_spacing = 8.0
number_of_slices = 13

# Create output directory
if not os.path.exists(out_path):
    os.makedirs(out_path, exists=True)

# Read CT image
ct_data, ct_affine, pixdim = fn.readFromNIFTI(ct_path)
img = nib.load(ct_path)
size = ct_data.shape
plane_size = size[0]

# Calculate position of LV/RV/Aorta
spatial_info = fn.calculate_spatial_information(ct_data, ct_affine, labels['LV'], labels['RV'], labels["Aorta"])

# Grabbing centroid and normal for all views
sa_normal_origin, la_2ch_normal_origin, la_3ch_normal_origin, la_4ch_normal_origin = fn.get_view_normal_origin(spatial_info)


def get_plane_vector(normal):
    if abs(normal[0]) < abs(normal[1]):
        return np.array([1, 0, 0])  # Align with X-axis
    else:
        return np.array([0, 1, 0])  # Align with Y-axis


# Create data
sa_data, sa_affine = fn.generate_scan_slices(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, 
                                             ct_data, ct_affine, 5, out_of_plane_spacing, plotOn = False)
la_2ch_data, la_2ch_affine = fn.generate_scan_slices(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size, 
                                                     ct_data, ct_affine, 5, out_of_plane_spacing, plotOn = False)
la_3ch_data, la_3ch_affine = fn.generate_scan_slices(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, 
                                                     ct_data, ct_affine, 5, out_of_plane_spacing, plotOn = False)
la_4ch_data, la_4ch_affine = fn.generate_scan_slices(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, 
                                                     ct_data, ct_affine, 5, out_of_plane_spacing, plotOn = False)

# sa_plane_vector = np.array([1, 0, 0])
# sa_data, sa_affine = fn.generate_scan_slices(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, 
#                                              ct_data, ct_affine, 1, out_of_plane_spacing, plane_vector=sa_plane_vector, plotOn=False)

# la_2ch_plane_vector = np.array([0, 1, 0])
# la_2ch_data, la_2ch_affine = fn.generate_scan_slices(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size, 
#                                                      ct_data, ct_affine, 1, out_of_plane_spacing, plane_vector=la_2ch_plane_vector, plotOn=False)

# la_3ch_plane_vector = np.array([1, 0, 0])
# la_3ch_data, la_3ch_affine = fn.generate_scan_slices(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, 
#                                                      ct_data, ct_affine, 1, out_of_plane_spacing, plane_vector=la_3ch_plane_vector, plotOn=False)

# la_4ch_plane_vector = np.array([0, 0, 1])
# la_4ch_data, la_4ch_affine = fn.generate_scan_slices(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, 
#                                                      ct_data, ct_affine, 1, out_of_plane_spacing, plane_vector=la_4ch_plane_vector, plotOn=False)


# Plot segmentation using plotly
fig = fn.show_segmentations(sa_data, sa_affine, fig=None)
fig = fn.show_segmentations(la_2ch_data, la_2ch_affine, fig=fig)
fig = fn.show_segmentations(la_3ch_data, la_3ch_affine, fig=fig)
fig = fn.show_segmentations(la_4ch_data, la_4ch_affine, fig=fig)
fig.show()

# # Add misalignment
# magnitude = 5

# Save views to nifti files
if not os.path.exists(out_path):
    os.makedirs(out_path, exists=True)

fn.save_Nifti(sa_data, sa_affine, spacing, out_of_plane_spacing, out_path + 'SA.nii.gz')
fn.save_Nifti(la_2ch_data, la_2ch_affine, spacing, out_of_plane_spacing, out_path + '2CH.nii.gz')
fn.save_Nifti(la_3ch_data, la_3ch_affine, spacing, out_of_plane_spacing, out_path + '3CH.nii.gz')
fn.save_Nifti(la_4ch_data, la_4ch_affine, spacing, out_of_plane_spacing, out_path + '4CH.nii.gz')

# fn.display_views(sa_data=sa_data, la_2CH_data=la_2ch_data, la_3CH_data=la_3ch_data, la_4CH_data=la_4ch_data)



# TODO: FIX MULTIPLE SLICES
# FIX GRID IN PLANE
# Update save nifit values, set zooms, set sform, check in Eidilon