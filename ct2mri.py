#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/10/30 16:22:52

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
import numpy as np
import functions as fn

# User inputs
ct_path = 'data/labels.nii.gz'
labels = {'LV': 1, 'RV': 2, 'Aorta': 3}
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

plt.figure(figsize=(6, 6))
plt.imshow(slice_data, cmap='gray', origin='lower')
plt.title("Short-Axis View")
plt.axis('off')
plt.show()

plane_size = size[0]
spacing = 1.0

print("Generating short axis view:\n")
fn.plot_short_axis(ct_data, ct_affine, lv_centroid, lv_long_axis, plane_size, spacing)

print("Generating 2-chamber view:\n")
fn.plot_2_chamber_view(ct_data, ct_affine, lv_centroid, rv_centroid, plane_size, spacing)

print("Generating 3-chamber view:\n")
fn.plot_3_chamber_view(ct_data, ct_affine, lv_centroid, aorta_centroid, plane_size, spacing)

print("Generating 4-chamber view:\n")
fn.plot_4_chamber_view(ct_data, ct_affine, lv_centroid, rv_centroid, lv_long_axis, plane_size, spacing)


