# -*- coding: utf-8 -*-

from __future__ import division
import time
import datetime
import sys
import numpy as np
import scipy
import os
import itertools

from random import randint
from random import choice
from collections import deque
from scipy.ndimage.interpolation import shift

from subprocess import call

import matplotlib.pyplot as plt

import config as cfg
import agregated_sea_consumption_v9 as mc
import plots_functions as pf
import triangular_plot as tp



''' version 0.9 of the code that explores under which parameter space the consumers need to consume algae in a costal desert'''
'''
returns an hysotgram of searesource consumed when varying several combinations of parameters
start with land-production vs number of consumers.
'''


def generate_grid(lim, which_par):

    consumers_parameter = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step )

    if which_par == 'land_productivity':
        change_par = np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
    elif which_par == 'high_land':
        change_par = np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
    elif which_par == 'tidal_deluge':
        change_par = np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
    elif which_par == 'high_sea':
        change_par = np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
    else:
        print ('wrong parameter name, names shall be: ')
        print ('land_productivity, high_land, tidal_deluge, high_sea')
        print ('you wrote', which_par)
        print ('exiting')
        sys.exit()

    return consumers_parameter, change_par

def generate_grids(lim, which_pars):

    change_pars = [None] * len(which_pars)
    for i, e in enumerate(which_pars):
        if e == 'land_productivity':
            change_pars[i] = np.arange(lim.min_land_prod, lim.max_land_prod, lim.prod_step)
        elif e == 'high_land':
            change_pars[i] = np.arange(lim.high_land_min, lim.high_land_max, lim.Lhigh_step)
        elif e == 'tidal_deluge':
            change_pars[i] = np.arange(lim.min_tidal_deluge, lim.max_tidal_deluge, lim.tidal_deluge_step)
        elif e == 'high_sea':
            change_pars[i] = np.arange(lim.high_sea_min, lim.high_sea_max, lim.high_sea_step)
        elif e == 'burners_number':
            change_pars[i] = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step )
        else:
            print ('wrong parameter name, names shall be: ')
            print ('land_productivity, high_land, tidal_deluge, high_sea, burners_number')
            print ('you wrote', which_pars)
            print ('exiting')
            sys.exit()

    return change_pars

def settings_mobility_seaFuels(par, lim):

    consumers_parameter = np.arange(lim.min_consumers, int(par.length/2), lim.con_step )

    low_land_mediumNhigh_sea = [lim.scarce_land_prod, lim.medium_tidal_deluge, lim.high_tidal_deluge]

    return consumers_parameter, low_land_mediumNhigh_sea


def call_model(par, consumers_number,  chang_par, which_par):#land_productivity ):
    #sim = mc.constants()

    #high_land, high_sea, land_vector, sea_vector = mc.create_n_inisialise_landscapes(par)
    par = cfg.par()
    par.n_consumers = consumers_number
    
    if which_par == 'land_productivity':
        par.land_productivity = chang_par
    elif which_par == 'high_land':
        par.high_land = chang_par
    elif which_par == 'tidal_deluge': 
        par.tidal_deluge = chang_par
    elif which_par == 'high_sea':    
        par.high_sea = chang_par
    elif which_par == 'burners_number':    
        par.n_consumers = chang_par

    norm_burned_land, norm_burned_sea, norm_movements, norm_rad = mc.main(par)
    #consumers_array = np.random.randint(low =  0, high = par.length, size = consumers_number)#initial positions of the consumers
    #all_sea_consumption = mc.acumulated_sea_consumption(par, sea_consumption)
    #all_jumps = mc.acumulated_n_jumps(par, n_jumps_array)
    
    return norm_burned_land, norm_burned_sea, norm_movements, norm_rad

def call_models(chang_pars, which_pars):#land_productivity ):
    
    for c, w in zip(chang_pars, which_pars):
        par = cfg.par()
        if w == 'land_productivity':
            par.land_productivity = c
        elif w == 'high_land':
            par.high_land = c
        elif w == 'tidal_deluge': 
            par.tidal_deluge = c
        elif w == 'high_sea':    
            par.high_sea = c
        elif w == 'burners_number':
            par.n_consumers = c
        else:
            print ('wrong parameter name, names shall be: ')
            print ('land_productivity, high_land, tidal_deluge, high_sea')
            print ('you wrote', which_pars)
            print ('exiting')
            sys.exit()

    norm_burned_land, norm_burned_sea, norm_movements, norm_rad = mc.main(par)
    return norm_burned_land, norm_burned_sea, norm_movements, norm_rad


def run_the_grids(change_pars, which_pars):

    sea_consumption_matrix  = np.zeros( (len(change_pars[0]), len(change_pars[1])  ) ) 
    land_consumption_matrix  = np.zeros( (len(change_pars[0]), len(change_pars[1])  ) ) 
    movements = np.zeros((len(change_pars[0]), len(change_pars[1])  )) 
    radius = np.zeros((len(change_pars[0]), len(change_pars[1])  ))
    
    
    for j, c1 in enumerate(change_pars[0]):
        print ('we are in gridssss', j, c1) 
        for k, c2 in enumerate(change_pars[1]):
            l, s, m, r = call_models( [c1, c2], which_pars)
            sea_consumption_matrix[j,k] = s
            land_consumption_matrix[j,k] = l
            movements[j,k] = m
            radius[j,k] = r

    return sea_consumption_matrix, land_consumption_matrix, movements, radius

def run_the_grid(par, consumers_parameter, change_par, which_par):# productivity_parameter

    sea_consumption_matrix  = np.zeros( (len(consumers_parameter), len(change_par)  ) ) 
    land_consumption_matrix  = np.zeros( (len(consumers_parameter), len(change_par)  ) ) 
    movements = np.zeros((len(consumers_parameter), len(change_par)  )) 
    radius = np.zeros((len(consumers_parameter), len(change_par)  ))
    for i, c in enumerate(consumers_parameter):
        
        for k, p in enumerate(change_par):
            
            l, s, m, r = call_model( par, c, p, which_par)
            print ('we are in', i, k, 'and we got ', l, s)
            sea_consumption_matrix[i,k] = s
            land_consumption_matrix[i,k] = l
            movements[i,k] = m
            radius[i,k] = r
    
    return sea_consumption_matrix, land_consumption_matrix, movements, radius

def run_one_value(par, value, consumers_parameter, which_par):# productivity_parameter

    sea_consumption_vectors  = np.zeros( len(consumers_parameter) ) 
    land_consumption_vectors  = np.zeros( len(consumers_parameter) ) 
    movements_vectors = np.zeros( len(consumers_parameter) ) 
    radius_vectors = np.zeros( len(consumers_parameter) )
    
    for i, c in enumerate(consumers_parameter):
        
        l, s, m, r = call_model( par, c, value, which_par)
        #print ('we are in', i, k, c, v, 'and we got ', m)
        sea_consumption_vectors[i] = s
        land_consumption_vectors[i] = l
        movements_vectors[i] = m
        radius_vectors[i] = r
    
    return movements_vectors, radius_vectors

def one_vector_runs(par, lim, value):# productivity_parameter
    
    vector_jumps_runs = []
    vector_radius_runs = []
    runs = par.runs
    consumers_parameters = np.arange(lim.min_consumers, lim.max_consumers, lim.con_step )
    par.land_productivity = lim.scarce_land_prod
    for i in range(runs):
        print ('\n run number', i, '\n')
        vectors_jumps, vectors_radius = run_one_value(par, value, consumers_parameters, 'tidal_deluge')
        vector_jumps_runs.append(vectors_jumps)
        vector_radius_runs.append(vectors_radius)


    # Calculate the mean and standard deviation
    mean_jump_values = np.mean(np.array(vector_jumps_runs), axis=0)
    std_dev_jump = np.std(np.array(vector_jumps_runs), axis=0)
    min_values_jump = np.min(np.array(vector_jumps_runs), axis=0)
    max_values_jump = np.max(np.array(vector_jumps_runs), axis=0)

    mean_range_values = np.mean(np.array(vector_radius_runs), axis=0)
    std_dev_ranges = np.std(np.array(vector_radius_runs), axis=0)
    min_values_ran = np.min(np.array(vector_radius_runs), axis=0)
    max_values_ran = np.max(np.array(vector_radius_runs), axis=0)


    ranges_stuff = dict(mean=mean_range_values, std=std_dev_ranges, min=min_values_ran, max=max_values_ran)
    jumps_stuff = dict(mean=mean_jump_values, std=std_dev_jump, min=min_values_jump, max=max_values_jump)
    
    #jumps_stuff = [mean_jump_values, std_dev_jump, min_values_jump, max_values_jump]
    return  jumps_stuff, ranges_stuff

def name_files(par, lim, which_pars):

    changes = ''

    for e in which_pars:
        if e == 'land_productivity':
            change = 'LpRan-' + str(lim.min_land_prod) + '-' + str(lim.max_land_prod)
        elif e == 'high_land':
            change = 'LhRan-' + str(lim.high_land_min) + '-' + str(lim.high_land_max)
        elif e == 'tidal_deluge':  
            change = 'SdRan-' + str(lim.min_tidal_deluge) + '-' + str(lim.max_tidal_deluge)
        elif e == 'high_sea':
            change = 'LdRan-' + str(lim.high_sea_min) + '-' + str(lim.high_sea_max)
        elif e == 'burners_number':
            change = 'nbRan-' + str(lim.min_consumers) + '-' + str(lim.max_consumers)
        else:
            print ('wrong parameter name, names shall be: land_productivity, high_land, tidal_deluge, high_sea')
            print ('you wrote', which_pars)
            print ('exiting')
            sys.exit()
        changes +=  change + '_' 

    name = 'quadMat' + '_' + changes + 'time' + str(par.time) +  '_len' + str(par.length) + '_R' + str(par.radius) +  '.npy'
    return name

def run_n_save_quadMats(lim, par, name, which_pars):   

    change_pars  = generate_grids(lim, which_pars)
    matrix_sea_consumption, matrix_land_consumption, matrix_jumps, matrix_radius =\
          run_the_grids(change_pars, which_pars)
    Mats = [matrix_sea_consumption, matrix_land_consumption, matrix_jumps, matrix_radius/par.length]
    
    save_name = name_files(par, lim, which_pars)
    
    if not os.path.exists(par.data_dir):
        os.makedirs(par.data_dir)

    np.save(par.data_dir + name + '_' + save_name, Mats)
    print('file saved in', par.data_dir + '_' + name + '_' + save_name)

def run_n_save_all_pairs(lim, par, name):

    all_pairs = (itertools.combinations(par.par_names, 2))
    for pair in all_pairs:
        print ('we are in pair', list(pair))
        run_n_save_quadMats(lim, par, name, list(pair))



    
#python scripts/resource_movement_grids.py name tidal_deluge
def main():
    
    start  = time.perf_counter() 

    name = sys.argv[1]

    lim = cfg.limits()
    par = cfg.par()

    if not os.path.exists(par.plots_dir):
        os.makedirs(par.plots_dir)
    
    consumers_parameter, change_par  = generate_grid(lim, par.which_par)
    matrix_sea_consumption, matrix_land_consumption, matrix_jumps, matrix_radius = run_the_grid(par, consumers_parameter, change_par, par.which_par)
    Mats = [matrix_sea_consumption, matrix_land_consumption, matrix_jumps, matrix_radius/par.length]
    pf.quad_plotSeaLandJumps(par, lim, Mats, name, par.which_par)
    pf.plot_land_resources_used(par, lim, Mats[1], name, par.which_par)
    
    which_pars = ['burners_number', 'high_land' ]#
    run_n_save_quadMats(lim, par, name, which_pars)
    #run_n_save_all_pairs(lim, par, name)
    save_name = name_files(par, lim, which_pars)
    Mats = np.load(par.data_dir + name + '_' + save_name, allow_pickle=True)
    pf.quad_plotSeaLandJumps(par, lim, Mats, name, par.which_par)
   
    #pf.quad_plotSeaLandJumps(par, lim, Mats, name, which_pars[1])

    

    tp.triangle_plotSeaLandJumps(par, lim, name, 'TF')
    
    ##jDic1, Rdic1 = one_vector_runs(par, lim, lim.medium_tidal_deluge)
    ##jDic2, Rdic2 = one_vector_runs(par, lim, lim.high_tidal_deluge)
    #pf.plot_2jumps_vectors(par, lim, consumers_parameter, vectors_jumps, name)
    #pf.plot_dist_2radius_vectors(par, lim, [ meanRanges1,  meanRanges2], [stdRanges1, stdRanges2], name)
    ##pf.plot_envelope_2jumps_vectors(par, lim, [ jDic1['mean'], jDic2['mean'] ], [jDic1['min'], jDic2['min']], [jDic1['max'], jDic2['max']], name)
    ##pf.plot_envelope_2radius_vectors(par, lim, [ Rdic1['mean'], Rdic2['mean'] ], [Rdic1['min'], Rdic2['min']], [Rdic1['max'], Rdic2['max']], name)

    print ('\n time to run all this stuff', str(datetime.timedelta(seconds = (time.perf_counter() - start))), '\n')
    
    plt.show()

if __name__ == "__main__":  
    main()