#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:14:40 2017

@author: sarkissi
"""
import numpy
import scipy.optimize as optim
import astra
    
class pdm(object):

    
    """ Algorithm class for Projection Distance Minimization (PDM) using global thresholding. 
	%
	% NOTATION: 
	% tau: thresholds values
	% rho: grey levels
    """
    # Public stuff:
    '''max_evals = 100
    meta = []
    display = []
    analyse = []
    process = []
    reconstruction = []
    data = [] 

			
    # Private:
    _wisdom_status = 1    
    '''
    def __init__(self):
        self.max_evals  = 100           # SETTING: maximum number of functions evaluations
        self.display    = 'off'         # SETTING: output of fminsearch optimization {'iter', 'off', 'on'}
        self.tau_fixed  = None           	# SETTING: boolean vector to determine which values of tau_initial should not be estimated
        self.rho_fixed  = None			    # SETTING: boolean vector to determine which values of rho_initial should not be estimated
        self.rho_method = 'fminsearch'	# SETTING: how to optimize the grey levels {'fminsearch','quadratic'}
        self.tau_method = 'hard'
        
        self.proj_geom	     = []			# REQUIRED: tomography object
        self.vol_geom      = []
        self.projections   = []			# REQUIRED: projections object, must match proj_geom of tomography
        self.projector     = 'FP3D_CUDA'
        self.astra_alg_id = None
        
        self.initialized		= False				# Is the object initialized?

        self.tau_initial = None
        self.rho_initial = None

    #def init_astra(self, volume, angles):
    #    sz = self.projections.shape
    #    self.proj_geom = astra.create_proj_geom('parallel', 1.0, sz[0], angles)
    #    self.vol_geom = astra.create_vol_geom(volume.shape)


    #------------ Perform global thresholding PDM
    # [S, rho_opt, tau_opt] = pdm.run(volume)
    def run(self, volume, angles, rho_initial, tau_initial):
        # Initialization
        if not self.initialized:
            self.initialize()
        
        
        self.tau_initial = numpy.array(tau_initial, dtype=numpy.float32)
        self.rho_initial = numpy.array(rho_initial, dtype=numpy.float32)
        
        if (self.tau_fixed is None) or (self.tau_fixed.shape != self.tau_initial.shape):
            self.tau_fixed = numpy.zeros_like(tau_initial, dtype=numpy.byte)
            
        if (self.rho_fixed is None) or (self.rho_fixed.shape != self.rho_initial.shape):
            self.rho_fixed = numpy.zeros_like(rho_initial, dtype=numpy.byte)
        
        # TODO: check volume to tomography.vol_geom
        #self.init_astra(volume, angles)
        
        # Create initial partition map
        pm = self.create_partition_mask(volume, tau_initial)
        
        # Remove fixed tau's
        tau_initial_variable = self.tau_initial[self.tau_fixed == 0]
       
        
        # Initialise astra variables
        fp = numpy.zeros((len(tau_initial) + 1, self.projections.shape[0]), dtype = numpy.float32)
        proj = numpy.zeros(astra.functions.geom_size(self.proj_geom), dtype=numpy.float32)
        bin_vol = numpy.zeros(astra.functions.geom_size(self.vol_geom), dtype = numpy.float32)
        
        vol_id = astra.data3d.link('-vol', self.vol_geom, bin_vol)
        sino_id = astra.data3d.link('-sino', self.proj_geom, proj)
        cfg = astra.astra_dict(self.projector)
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        self.astra_alg_id = astra.algorithm.create(cfg)
    
        
        # Optimization of tau
        #options = optimset('display', this.display, 'MaxFunEvals', this.max_evals);
        if self.tau_method == 'hard':
            tau_opt = optim.fmin(func=self.optim_func_tau_hard, x0=tau_initial_variable, args=(volume, pm, fp, proj, bin_vol,))
        elif self.tau_method == 'soft':
            tau_opt = self.newton()
        else:
            raise ValueError('Invalid value for option "rho_method"')
        # Optimization of rho
        [_, rho_opt] = self.optim_func_tau_hard(tau_opt, volume, pm, fp, proj, bin_vol, full_output = True)
        
        
        # Delete astra variables
        astra.algorithm.delete(self.astra_alg_id)
        astra.data3d.delete([vol_id, sino_id])
        
        # Insert fixed tau's
        tmp = tau_opt.copy()
        tau_opt = self.tau_initial.copy()
        tau_opt[self.tau_fixed == 0] = tmp
        
        # Segmentation
        #print(tau_opt)
        pm = self.create_partition_mask(volume, tau_opt)
        seg = numpy.zeros(pm.shape, dtype = numpy.float32)
        
        
        #print(rho_opt)
        
        for i in range(0,len(rho_opt)):
            seg[pm == i] = rho_opt[i]

        return [seg, rho_opt, tau_opt]
    
    
    #--------------- Initialize 
    def initialize(self):
        # TODO: check tomography.proj_geom vs projections
        if len(self.projections.shape) > 1:
            self.projections = self.projections.ravel()
        
        self.initialized = True
        return self.initialized
    
    #--------------- Optimization function for tau
    def optim_func_tau_hard(self, tau, volume, pm, fp, proj, bin_vol, full_output=False):
        
        # Insert fixed tau's
        tau_all = self.tau_initial.copy()
        tau_all[self.tau_fixed == 0] = tau
        
        # Create partition mask
        self.create_partition_mask(volume, tau_all, pm)

        # Create forward projection of each partition            
        for i in range(0, len(tau_all) + 1):
            #print('pm == i')
            #print(numpy.sum(pm == i))
            bin_vol[:] = pm == i
            #print(numpy.sum(bin_vol))
            astra.algorithm.run(self.astra_alg_id,1)
            #print('proj sum', numpy.sum(proj))
            fp[i,:] = proj.ravel()
            
        
        # Find rho
        if (self.rho_method == 'fminsearch'):
            [rho_opt, proj_diff] = self.rho_fminsearch_opt(fp)
        elif (self.rho_method == 'quadratic'):
            [rho_opt, proj_diff] = self.rho_quadratic_opt(fp)
        else:
            raise ValueError('Invalid value for option "rho_method"')
        
        #print(proj_diff)
        if full_output:
            return proj_diff, rho_opt
        else:
            return proj_diff

    #--------------- Optimization function for rho using fmin
    def rho_fminsearch_opt(self, fp):
        
        # Remove fixed rho's
        rho_initial_variable = self.rho_initial[self.rho_fixed == 0]
                                
        # options = optimset('display', 'off', 'MaxFunEvals', 50);
        rho_opt, proj_diff, _,_,_ = optim.fmin(func=self.optim_func_rho, x0=rho_initial_variable, args=(fp,), full_output=True)

        # Insert fixed rho's
        tmp = rho_opt.copy()
        rho_opt = self.rho_initial.copy()
        rho_opt[self.rho_fixed == 0] = tmp
            
        return [rho_opt, proj_diff]

    #--------------- Optimization function for rho using quadratic optimization
    def rho_quadratic_opt(self, fp):
        # Todo: fixed rho's
        
        # c
        c = fp.dot(self.projections)
        #c = numpy.zeros(fp.shape[0], dtype = numpy.float32)
        #for i in range(0, len(fp)):
            #c[i] = numpy.sum(-2.0 * self.projections * fp[i])
            
        # Q
        Q = fp.dot(fp.T)
        
        #print('Q', Q)
        
        rho_opt = - numpy.linalg.pinv(Q).dot(c)
        #print('rho', rho_opt)
        
        proj_diff = self.optim_func_rho(rho_opt, fp)
        return [rho_opt, proj_diff]

    #--------------- Projection difference function for a given rho and a set of forward projections
    def optim_func_rho(self, rho, fp):
			
        # Insert fixed rho's
        tmp = rho.copy()
        rho = self.rho_initial.copy()
        rho[self.rho_fixed == 0] = tmp

        residu = self.projections
        for i in range(0, len(rho)):
            residu = residu - rho[i] * fp[i]

        s = residu * residu
        return numpy.sum(s)




    #--------------- Create partition mask
    def create_partition_mask(self, volume, tau, pm = None):
        ret = True
        
        # Sort tau to ensure increasing values
        tau.sort()
        
        if pm is None:
            pm = numpy.zeros_like(volume, dtype = numpy.uint8)
        else:
            ret = False
           
        for i in range(0,len(tau)):
            pm[volume > tau[i]] = i+1
        
        if ret:
            return pm