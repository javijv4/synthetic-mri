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
inplane_spacing = 2.0       # Use this when generating the grid
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

# Create data
sa_data, sa_affine = fn.generate_scan_slices(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing)
la_2ch_data, la_2ch_affine = fn.generate_scan_slices(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing)
la_3ch_data, la_3ch_affine = fn.generate_scan_slices(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing)
la_4ch_data, la_4ch_affine = fn.generate_scan_slices(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing)

# Plot segmentation using plotly
fig = fn.show_segmentations(sa_data, sa_affine)
fig = fn.show_segmentations(la_2ch_data, la_2ch_affine, fig=fig)
fig = fn.show_segmentations(la_3ch_data, la_3ch_affine, fig=fig)
fig = fn.show_segmentations(la_4ch_data, la_4ch_affine, fig=fig)
fig.show()
