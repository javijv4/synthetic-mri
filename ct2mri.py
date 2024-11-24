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


# User inputs
ct_path = 'data/labels.nii.gz'
out_path = 'data/'
labels = {'LV': 1, 'RV': 3, 'Aorta': 6}
inplane_spacing = 1.0       # Use this when generating the grid

# Read CT image
ct_data, ct_affine, pixdim = fn.readFromNIFTI('data/labels.nii.gz')
size = ct_data.shape

# Calculate position of LV/RV/Aorta
spatial_info = fn.calculate_spatial_information(ct_data, ct_affine)

print("\nSpatial Information:")
for key, value in spatial_info.items():
    print(f"{key}: {value}")

lv_centroid = spatial_info["LV Centroid"]
rv_centroid = spatial_info["RV Centroid"]
aorta_centroid = spatial_info["Aorta Centroid"]
lv_long_axis = spatial_info["LV Long Axis"]

# print("LV Centroid:", lv_centroid)
# print("RV Centroid:", rv_centroid)
# print("Aorta Centroid:", aorta_centroid)
# print("LV Long Axis:", lv_long_axis)

# Generating short-axis slices # TODO make this for multiple slices along the long axis
origin = lv_centroid
normal = lv_long_axis

print("\nGenerating grid and interpolating slice...")
xyz = fn.grid_in_plane(origin, normal, 50, size[0])
slice_data = fn.interpolate_image(xyz, ct_data, ct_affine)

# plt.figure(figsize=(6, 6))
# plt.imshow(slice_data, cmap='gray', origin='lower')
# plt.title("Short-Axis View")
# plt.axis('off')
# plt.show()

plane_size = size[0]
spacing = 1.0
out_of_plane_spacing = 8.0
number_of_slices = 13

# # Find the affine for each view given the normal and center of the slice
# sa_affine = ... # 4x4 affine matrix
# la_2CH_affine = ... # 4x4 affine matrix
# la_3CH_affine = ... # 4x4 affine matrix
# la_4CH_affine = ... # 4x4 affine matrix


# print("\nGenerating interactive short axis view...")
fn.plot_interactive_view(ct_data, ct_affine, lv_centroid, lv_long_axis, plane_size=100, spacing=1.0, num_slices=20)

print("\nGenerating short axis view:")
sa_data, _ = fn.plot_short_axis(ct_data, ct_affine, lv_centroid, lv_long_axis, plane_size,
                                spacing, out_of_plane_spacing, number_of_slices, plotOn = False)

print("\nGenerating 2-chamber view:")
la_2CH_data, _ = fn.main_2chamber_view(ct_path, plane_size, spacing, plotOn = False)

print("\nGenerating 3-chamber view:")
la_3CH_data, _ = fn.main_3chamber_view(ct_path, plane_size, spacing, plotOn = False)

print("\nGenerating 4-chamber view:")
la_4CH_data, _ = fn.main_4chamber_view(ct_path, plane_size, spacing, plotOn = False)

# TODO need to figure out a way to show all 4 views at once without having to close out the tab everytime


# Add misalignment
magnitude = 5





# Save views to nifti files
if not os.path.exists(out_path):
    os.makedirs(out_path, exists=True)


print("\nDisplaying all views...")
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Create a 2x2 grid of subplots

# Short-Axis View
if sa_data is None:
    print("Failed to generate short-axis view.")
else:
    # Use only the slice_data for imshow
    axes[0, 0].imshow(sa_data, cmap='gray', origin='lower')
    axes[0, 0].set_title("Short-Axis View")
    axes[0, 0].axis('off')

# 2-Chamber View
if sa_data is None:
    print("Failed to generate 2-chamber view.")
else:
    axes[0, 1].imshow(la_2CH_data, cmap='gray', origin='lower')
    axes[0, 1].set_title("2-Chamber View")
    axes[0, 1].axis('off')

# 3-Chamber View
if sa_data is None:
    print("Failed to generate 3-chamber view.")
else:
    axes[1, 0].imshow(la_3CH_data, cmap='gray', origin='lower')
    axes[1, 0].set_title("3-Chamber View")
    axes[1, 0].axis('off')

# 4-Chamber View
if la_4CH_data is None:
    print("Failed to generate 4-chamber view.")
else:
    axes[1, 1].imshow(la_4CH_data, cmap='gray', origin='lower')
    axes[1, 1].set_title("4-Chamber View")
    axes[1, 1].axis('off')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()



