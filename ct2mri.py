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
inplane_spacing = 1.0       # Use this when generating the grid
spacing = 2.0       # Use this when generating the grid
out_of_plane_spacing = 8.0
number_of_slices = 13
# Create output directory
if not os.path.exists(out_path):
    os.makedirs(out_path, exists=True)

# Read CT image
ct_data, ct_affine, pixdim = fn.readFromNIFTI('data/labels.nii.gz')
size = ct_data.shape
plane_size = size[0]

# Calculate position of LV/RV/Aorta
spatial_info = fn.calculate_spatial_information(ct_data, ct_affine, labels['LV'], labels['RV'], labels["Aorta"])

print("\nSpatial Information:")
for key, value in spatial_info.items():
    print(f"{key}: {value}")

lv_centroid = spatial_info["LV Centroid"]
rv_centroid = spatial_info["RV Centroid"]
aorta_centroid = spatial_info["Aorta Centroid"]
lv_long_axis = spatial_info["LV Long Axis"]

g1, _ = fn.grid_in_plane(lv_centroid, lv_long_axis, inplane_spacing, 2)
g2 = fn.grid_in_plane2(lv_centroid, lv_long_axis, inplane_spacing, 2)
print(g1[0:3])
print(g2[0:3])
# print("LV Centroid:", lv_centroid)
# print("RV Centroid:", rv_centroid)
# print("Aorta Centroid:", aorta_centroid)
# print("LV Long Axis:", lv_long_axis)
plane_size = size[0]
spacing = 1.0
out_of_plane_spacing = 8.0
number_of_slices = 13
# # # Find the affine for each view given the normal and center of the slice
# # sa_affine = ... # 4x4 affine matrix
# # la_2CH_affine = ... # 4x4 affine matrix
# # la_3CH_affine = ... # 4x4 affine matrix
# # la_4CH_affine = ... # 4x4 affine matrix
# # print("\nGenerating interactive short axis view...")
# fn.plot_interactive_view(ct_data, ct_affine, lv_centroid, lv_long_axis, plane_size=100, spacing=1.0, num_slices=20)
print("\nGenerating short axis view:")
sa_data, saAffineData = fn.plot_short_axis(ct_data, ct_affine, lv_centroid, lv_long_axis, plane_size,
                                spacing, out_of_plane_spacing, number_of_slices, plotOn = False)

print("\nGenerating 2-chamber view:")
la_2CH_data, la_2CH_affine = fn.main_2chamber_view(ct_path, plane_size, spacing, out_of_plane_spacing, number_of_slices, plotOn = False)
la_2CH_data, la_2CH_affine = fn.main_2chamber_view(ct_path, plane_size, spacing, out_of_plane_spacing, 1, plotOn = False)

print("\nGenerating 3-chamber view:")
la_3CH_data, la_3CH_affine = fn.main_3chamber_view(ct_path, plane_size, spacing, out_of_plane_spacing, number_of_slices, plotOn = False)
la_3CH_data, la_3CH_affine = fn.main_3chamber_view(ct_path, plane_size, spacing, out_of_plane_spacing, 1, plotOn = False)

print("\nGenerating 4-chamber view:")
la_4CH_data, la_4CH_affine = fn.main_4chamber_view(ct_path, plane_size, spacing, out_of_plane_spacing, number_of_slices, plotOn = False)
la_4CH_data, la_4CH_affine = fn.main_4chamber_view(ct_path, plane_size, spacing, out_of_plane_spacing, 1, plotOn = False)
fn.display_views(sa_data=sa_data, la_2CH_data=la_2CH_data, la_3CH_data=la_3CH_data, la_4CH_data=la_4CH_data)

# Save views to nifti files
fn.save_Nifti(sa_data, saAffineData, out_path + 'sa.nii.gz')
fn.save_Nifti(la_2CH_data, la_2CH_affine, out_path + 'la_2CH.nii.gz')
fn.save_Nifti(la_3CH_data, la_3CH_affine, out_path + 'la_3CH.nii.gz')
fn.save_Nifti(la_4CH_data, la_4CH_affine, out_path + 'la_4CH.nii.gz')

# Plot segmentation using plotly
fig = fn.show_segmentations(la_2CH_data, la_2CH_affine)
fig = fn.show_segmentations(la_3CH_data, la_3CH_affine, fig=fig)
fig = fn.show_segmentations(la_4CH_data, la_4CH_affine, fig=fig)
fig.show()

# # Add misalignment
# magnitude = 5

# # Save views to nifti files
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exists=True)
fn.display_views(sa_data=sa_data, la_2CH_data=la_2CH_data, la_3CH_data=la_3CH_data, la_4CH_data=la_4CH_data)
