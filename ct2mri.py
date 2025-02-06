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
paths = {
    'clean': {
        'data': 'data/clean/',
        'bvg': '../../bvmodelgen_urop/data/clean/'
    },
    'misaligned': {
        'data': 'data/misaligned/',
        'bvg': '../../bvmodelgen_urop/data/misaligned/'
    }
}

labels = {'LV': 1, 'RV': 3, 'Aorta': 6}
inplane_spacing = 3.0       # Use this when generating the grid
spacing = 2.0       # Use this when generating the grid
out_of_plane_spacing = 10.0
number_of_slices = 13

# Create output directory
if not os.path.exists(paths['clean']['data']):
    os.makedirs(paths['clean']['data'], exists=True)

# Read CT image
ct_data, ct_affine, pixdim = fn.readFromNIFTI(ct_path)
img = nib.load(ct_path)
size = ct_data.shape
plane_size = size[0]

# Calculate position of LV/RV/Aorta
spatial_info = fn.calculate_spatial_information(ct_data, ct_affine, labels['LV'], labels['RV'], labels["Aorta"])

# Grabbing centroid and normal for all views
sa_normal_origin, la_2ch_normal_origin, la_3ch_normal_origin, la_4ch_normal_origin = fn.get_view_normal_origin(spatial_info)

# Create data
sa_data, sa_affine, sa_data_misaligned = fn.generate_scan_slices(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, 
                                             ct_data, ct_affine, 13, out_of_plane_spacing, plotOn = False)
la_2ch_data, la_2ch_affine, la_2ch_data_misaligned = fn.generate_scan_slices(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size, 
                                                     ct_data, ct_affine, 1, out_of_plane_spacing, plotOn = False)
la_3ch_data, la_3ch_affine, la_3ch_data_misaligned = fn.generate_scan_slices(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, 
                                                     ct_data, ct_affine, 1, out_of_plane_spacing, plotOn = False)
la_4ch_data, la_4ch_affine, la_4ch_data_misaligned = fn.generate_scan_slices(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, 
                                                     ct_data, ct_affine, 1, out_of_plane_spacing, plotOn = False)

fig = fn.show_segmentations(sa_data, sa_affine, fig=None)
fig = fn.show_segmentations(la_2ch_data, la_2ch_affine, fig=fig)
fig = fn.show_segmentations(la_3ch_data, la_3ch_affine, fig=fig)
fig = fn.show_segmentations(la_4ch_data, la_4ch_affine, fig=fig)

fn.save_all_nifti_files(sa_data, sa_affine, la_2ch_data, la_2ch_affine, 
                        la_3ch_data, la_3ch_affine, la_4ch_data, la_4ch_affine, sa_data_misaligned, 
                         la_2ch_data_misaligned, la_3ch_data_misaligned, la_4ch_data_misaligned, 
                         spacing, out_of_plane_spacing, paths)

truth_endpoints = fn.display_views(paths, 'clean', la_2CH_data= la_2ch_data, la_3CH_data=la_3ch_data, la_4CH_data= la_4ch_data, la_2CH_affine=la_2ch_affine, la_3CH_affine=la_3ch_affine, la_4CH_affine=la_4ch_affine)
misalgined_endpoints = fn.display_views(paths, 'misaligned', la_2CH_data= la_2ch_data_misaligned, la_3CH_data=la_3ch_data_misaligned, la_4CH_data= la_4ch_data_misaligned, la_2CH_affine=la_2ch_affine, la_3CH_affine=la_3ch_affine, la_4CH_affine=la_4ch_affine)

# TODO: Create folders for them for foreign users
# TODO: mark the ventricles in the heart
# TODO: create a 2D mask thats 1 on the points where the ventricles are / find valve masks

# TODO: find valves for misaligned and aligned. and then save as nifti files 
# TODO: Update the BVG_ folder with all the saved elements and the valve locations
# TODO: Create individual functiions for 2ch 3ch and 4ch valve location DONE 
# TODO: experiemtn with differnt boundaries detecitons thick vs etc  // DONE and failed :(


'''
Thick Thick - what we had
Thick subpixel - no overlapping boundary
thick outher - way off
thick inner - way off

outer outer - way off
outer subpixel - no overlapping boundary
outer inner -way off

inner inner - no overlapping boundary
inner subpixel - no overlapping boundary

subpixel subpixel - no overlapping boundary

'''