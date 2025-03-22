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
ct_path = 'data/75_trimmed_label_maps.nii.gz'
paths = {
    'clean': {
        'data': 'data/clean/',
        'bvg' : '../../bvmodelgen_urop/data/clean/'
    },
    'misaligned': {
        'data': 'data/misaligned/Run2/',
        'bvg' : '../../bvmodelgen_urop/data/misaligned/Run1/'
    },
    'MRI': {
        'data': 'data/MRI/Run1/',
        'bvg' : '../../bvmodelgen_urop/data/MRI/Run1/'
    },
    'breath': {
        'data': 'data/breath/Run1/',
        'bvg' : '../../bvmodelgen_urop/data/breath/Run1/'
    }
}

labels = {'LV': 1, 'RV': 3, 'Aorta': 6}
inplane_spacing = 3.0       # Use this when generating the grid
spacing = 2.0       # Use this when generating the grid
out_of_plane_spacing = 10.0
number_of_slices = 13
breath_holding_error = 0.5
# misalignment = 8

# Create output directory
if not os.path.exists(paths['clean']['data']):
    os.makedirs(paths['clean']['data'])
if not os.path.exists(paths['misaligned']['data']):
    os.makedirs(paths['misaligned']['data'])
if not os.path.exists(paths['clean']['bvg']):
    os.makedirs(paths['clean']['bvg'])
if not os.path.exists(paths['misaligned']['bvg']):
    os.makedirs(paths['misaligned']['bvg'])
# Reaxd CT image
ct_data, ct_affine, pixdim = fn.readFromNIFTI(ct_path)
img = nib.load(ct_path)
size = ct_data.shape
plane_size = size[0]

# Calculate position of LV/RV/Aorta
spatial_info = fn.calculate_spatial_information(ct_data, ct_affine, labels['LV'], labels['RV'], labels["Aorta"])

# Grabbing centroid and normal for all views
sa_normal_origin, la_2ch_normal_origin, la_3ch_normal_origin, la_4ch_normal_origin = fn.get_view_normal_origin(spatial_info)
misalignment_array = [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8]
# Create data

misalignment_array = [8]
# for misalignment in np.arange(.9, .9 , 0.25):
for misalignment in misalignment_array:
    print("Currently Processing level", misalignment, " misalignment")    
    # sa_data, sa_affine = fn.generate_scan_slices_main(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 13, out_of_plane_spacing, plotOn=False)
    # la_2ch_data, la_2ch_affine = fn.generate_scan_slices_main(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size,  ct_data, ct_affine, 1, out_of_plane_spacing, plotOn=False)
    # la_3ch_data, la_3ch_affine = fn.generate_scan_slices_main(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, plotOn=False)
    # la_4ch_data, la_4ch_affine = fn.generate_scan_slices_main(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, plotOn=False)

    # sa_data_misaligned = fn.generate_scan_slices_misaligned(sa_data, sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 13, out_of_plane_spacing, misalignment)
    # la_2ch_data_misaligned = fn.generate_scan_slices_misaligned(la_2ch_data, la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, misalignment)
    # la_3ch_data_misaligned = fn.generate_scan_slices_misaligned(la_3ch_data, la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, misalignment)
    # la_4ch_data_misaligned = fn.generate_scan_slices_misaligned(la_4ch_data, la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, misalignment)
    
    sa_data_breath, sa_affine_breath = fn.generate_scan_slices_breathHolding(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 13, out_of_plane_spacing, False, misalignment)
    la_2ch_data_breath, la_2ch_affine_breath = fn.generate_scan_slices_breathHolding(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size,  ct_data, ct_affine, 1, out_of_plane_spacing, False, misalignment)
    la_3ch_data_breath, la_3ch_affine_breath = fn.generate_scan_slices_breathHolding(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, False, misalignment)
    la_4ch_data_breath, la_4ch_affine_breath = fn.generate_scan_slices_breathHolding(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, False, misalignment)

    # sa_data_MRIerror, sa_affine_MRIerror = fn.generate_scan_slices_MRIerror(sa_normal_origin[1], sa_normal_origin[0], inplane_spacing, plane_size,ct_data, ct_affine, 13, out_of_plane_spacing, False, misalignment)
    # la_2ch_data_MRIerror, la_2ch_affine_MRIerror = fn.generate_scan_slices_MRIerror(la_2ch_normal_origin[1], la_2ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, False, misalignment)
    # la_3ch_data_MRIerror, la_3ch_affine_MRIerror = fn.generate_scan_slices_MRIerror(la_3ch_normal_origin[1], la_3ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, False, misalignment)
    # la_4ch_data_MRIerror, la_4ch_affine_MRIerror = fn.generate_scan_slices_MRIerror(la_4ch_normal_origin[1], la_4ch_normal_origin[0], inplane_spacing, plane_size, ct_data, ct_affine, 1, out_of_plane_spacing, False, misalignment)

    # fn.display_segmentations(datasets = [sa_data, la_2ch_data, la_3ch_data, la_4ch_data], affines = [sa_affine, la_2ch_affine, la_3ch_affine, la_4ch_affine])
    # fn.display_segmentations(datasets = [sa_data_misaligned, la_2ch_data_misaligned, la_3ch_data_misaligned, la_4ch_data_misaligned], affines = [sa_affine, la_2ch_affine, la_3ch_affine, la_4ch_affine])
    # fn.display_segmentations(datasets = [sa_data_MRIerror, la_2ch_data_MRIerror, la_3ch_data_MRIerror, la_4ch_data_MRIerror], affines = [sa_affine_MRIerror, la_2ch_affine_MRIerror, la_3ch_affine_MRIerror, la_4ch_affine_MRIerror])
    fn.display_segmentations(datasets = [sa_data_breath, la_2ch_data_breath, la_3ch_data_breath, la_4ch_data_breath], affines = [sa_affine_breath, la_2ch_affine_breath, la_3ch_affine_breath, la_4ch_affine_breath])


    # fn.save_truth_filees(sa_data, sa_affine, la_2ch_data, la_2ch_affine, la_3ch_data, la_3ch_affine, la_4ch_data, la_4ch_affine, paths)
    # fn.save_error_files(sa_data_MRIerrorla_2ch_data_MRIerror, sa_affine, la_2ch_data_misaligned, la_2ch_affine, la_3ch_data_misaligned, la_3ch_affine, la_4ch_data_misaligned, la_4ch_affine, paths, misalignment, 'misaligned')
    # fn.save_error_files(sa_data_breath, sa_affine_breath, la_2ch_data_breath, la_2ch_affine_breath, la_3ch_data_breath, la_3ch_affine_breath, la_4ch_data_breath, la_4ch_affine_breath, paths, misalignment, 'breath')
    # fn.save_error_files(sa_data_MRIerror, sa_affine_MRIerror, la_2ch_data_MRIerror, la_2ch_affine_MRIerror, la_3ch_data_MRIerror, la_3ch_affine_MRIerror, la_4ch_data_MRIerror, la_4ch_affine_MRIerror, paths, misalignment, 'MRI')


    # truth_endpoints = fn.display_views(paths, 'clean', misalignment = 0, sa_data= sa_data, la_2CH_data= la_2ch_data, la_3CH_data=la_3ch_data, la_4CH_data= la_4ch_data, la_2CH_affine=la_2ch_affine, la_3CH_affine=la_3ch_affine, la_4CH_affine=la_4ch_affine, ct_affine= ct_affine)
    # misalgined_endpoints = fn.display_views(paths, 'misaligned', misalignment, sa_data= sa_data, la_2CH_data= la_2ch_data, la_3CH_data=la_3ch_data, la_4CH_data= la_4ch_data, la_2CH_affine=la_2ch_affine, la_3CH_affine=la_3ch_affine, la_4CH_affine=la_4ch_affine, ct_affine= ct_affine)
    # breath_endpoints = fn.display_views(paths, 'breath', misalignment, sa_data= sa_data_breath, la_2CH_data= la_2ch_data_breath, la_3CH_data=la_3ch_data_breath, la_4CH_data= la_4ch_data_breath, la_2CH_affine=la_2ch_affine_breath, la_3CH_affine=la_3ch_affine_breath, la_4CH_affine=la_4ch_affine_breath, ct_affine= ct_affine)
    # MRI_endpoints = fn.display_views(paths, 'MRI', misalignment, sa_data= sa_data_MRIerror, la_2CH_data= la_2ch_data_MRIerror, la_3CH_data=la_3ch_data_MRIerror, la_4CH_data= la_4ch_data_MRIerror, la_2CH_affine=la_2ch_affine_MRIerror, la_3CH_affine=la_3ch_affine_MRIerror, la_4CH_affine=la_4ch_affine_MRIerror, ct_affine= ct_affine)
