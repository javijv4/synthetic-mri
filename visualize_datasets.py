#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/11 09:54:18

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import functions as fn



path = 'data/MRI_Breath/Run1/'

sa_data, sa_affine, _ = fn.readFromNIFTI(path + 'SA_MRI_Breath_MRI1.00_BH1.00.nii.gz', correct_ras=False)
la_2ch_data, la_2ch_affine, _ = fn.readFromNIFTI(path + '2CH_MRI_Breath_MRI1.00_BH1.00.nii.gz', correct_ras=False)
la_3ch_data, la_3ch_affine, _ = fn.readFromNIFTI(path + '3CH_MRI_Breath_MRI1.00_BH1.00.nii.gz', correct_ras=False)
la_4ch_data, la_4ch_affine, _ = fn.readFromNIFTI(path + '4CH_MRI_Breath_MRI1.00_BH1.00.nii.gz', correct_ras=False)

# Correct for plotly plotting
sa_data = np.swapaxes(sa_data, 0, 1)
la_2ch_data = np.swapaxes(la_2ch_data, 0, 1)
la_3ch_data = np.swapaxes(la_3ch_data, 0, 1)
la_4ch_data = np.swapaxes(la_4ch_data, 0, 1)

fn.display_segmentations(datasets = [sa_data, la_2ch_data, la_3ch_data, la_4ch_data], affines = [sa_affine, la_2ch_affine, la_3ch_affine, la_4ch_affine])

