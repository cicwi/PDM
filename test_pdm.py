#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:09:26 2017

@author: kostenko & der sarkissian
"""
#%%
import sys
sys.path.append('/home/sarkissi/Development/tomo_box/')
import tomobox as tw
import os
import numpy
import time

#astra.log.setOutputScreen(astra.log.STDOUT, astra.log.DEBUG)

#%% Initialize:
projections = tw.sinogram()


#% Preprocess:
files_dir = '/export/scratch1/sarkissi/Hinf_24-13-23-SU1-T1' 
#files_dir = '/export/scratch2/sarkissi/Data/seashell/Projections/SU1/Hinf-AMT24-6a/Hinf-AMT24-6a-SU1-T4/Hinf-AMT24-6a-07-SU1-T4'

# Load projections and geometry
xcrop = 1500
ycrop = 800
projections.io.read_raw(path = os.path.join(files_dir, ''), x_range = [xcrop,4000-xcrop], y_range = [ycrop,2672-ycrop])
projections.io.read_meta(files_dir,kind='SkyScan')
projections.io.read_skyscan_thermalshifts(files_dir)
projections.meta.geometry['det_tilt'] = 0.0

projections.process.flat_field(kind = 'SkyScan')
projections.process.log()

volume = tw.read_image_stack(os.path.join(files_dir, 'recon', 'out_0000.tiff'))

# Initialise astra variables
projections.reconstruct._initialize_astra()

#%%
import pdm
segmenter = pdm.pdm()

# initialise pdm object
segmenter.rho_method = 'quadratic'
segmenter.proj_geom = projections.reconstruct.proj_geom
segmenter.vol_geom = projections.reconstruct.vol_geom
segmenter.projections = projections.data._data

tau_initial = [0.0,0.00001,0.0001]
rho_initial = [0.00001,0000.1,0.001,0.01]
segmenter.rho_fixed = numpy.array([False, False, False, False], dtype = numpy.bool)
segmenter.tau_fixed = numpy.array([False, False, False], dtype = numpy.bool)


#
#%% Run PDM
start = time.time()
[S, rho_opt, tau_opt] = segmenter.run(volume, projections.meta.theta, rho_initial, tau_initial)
end = time.time()
print(end - start)

#%%
v = tw.volume(S)
#%%
v.io.save(path = '', fname='seg', fmt = 'tiff')
