'''

'''

import matplotlib.pyplot as plt
import numpy as np
import os #I think we never use this?
import glob 
from astropy.io import fits
from pylab import * #idk if we use this, i know one of its functions was used in one of our previous programs butsince it's * idk
import juliet
import time

def STIS_dataset_from_fits(directory = "HST-STIS-HD189733b", visit = 1):
	"""
	Function for obtaining a dataset array from original Hubble STIS files.

	Parameters
	----------
	directory : string
		Directory of the dataset of interest as provided by MAST. Future
		implementations may allow direct querying from MAST.
	visit : integer
		Specific visit to be evaluated.

	Returns
	-------
	STIS_dataset : 3D numpy array
		Returns a numpy array with a time dimension [explain better]
	"""

	files = sorted(glob.glob(directory + '/visit' + str(visit) + '/*.fits'))

	hdul = fits.open(files[0])
	test_file_1_arr = hdul[0].data
	testing_array = np.zeros(test_file_1_arr.shape)
	for f in files:
	    hdul = fits.open(f)
	    testing_array = testing_array + hdul[0].data

	Ntimes = len(files)

	full_array = np.zeros((Ntimes,testing_array.shape[0],testing_array.shape[1]))
	date_array = np.zeros(Ntimes)

	for f in files:
	    data,h = fits.getdata(f,header=True)
	    full_array[int(f[33:36]),:,:] = data
	    tstart, tend = h['EXPSTART'], h['EXPEND']
	    date_array[int(f[33:36])] = (tstart+tend)/2

	date_array = np.reshape(date_array, (len(date_array),1))

	# WORK IN PROGRESS!!!!!

def get_mad_sigma(x):
    """
    Estimate the robust version of the standard deviation (sigma) using the Median Absolute Deviation (MAD).
    See: https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(x-np.median(x)))*1.4826

def cosmic_ray_correct(input_dataset):
    """
    Function for correcting cosmic rays from the STIS dataset.
    
    Parameter
    ---------
    input_dataset : 3D numpy array
        Input dataset, 3D numpy array with time as axis 0,
        spatial as axis 1, and wavelength as axis 2.
        Format used in this notebook.

    Returns
    -------
    input_dataset : 3D numpy array
        Corrected version of the input_dataset with cosmic rays
        removed and interpolated.
    """
    
    bool_idx_arr = np.zeros(input_dataset.shape, dtype=bool)
    
    # First step is to identify outliers
    # for i in range(input_dataset.shape[0]):  
    for (x,y), pxl in np.ndenumerate(input_dataset[0,:,:]):
        pxl_med = np.median(input_dataset[:,x,y])
        pxl_sigma = get_mad_sigma(input_dataset[:,x,y])
        pxl_cr_idx_bool = input_dataset[:,x,y] > pxl_med + 5 * pxl_sigma
        #print(np.sum(pxl_cr_idx_bool))
        #if np.sum(pxl_cr_idx_bool) >= 1:
        #   print(pxl_cr_idx_bool)
        bool_idx_arr[:,x,y] = pxl_cr_idx_bool
        
    # Now, we have identified the indexes where pixels are lost due to cosmic rays
    # We now need to interpolate them with their vertical counterparts

    work_dataset = np.copy(input_dataset)

    x = np.arange(work_dataset.shape[1])
    for (i,j), col in np.ndenumerate(work_dataset[:,0,:]):
        if np.sum(bool_idx_arr[i,:,j]):
            xp = x[(~bool_idx_arr[i,:,j]).nonzero()]
            fp = work_dataset[i,:,j][(~bool_idx_arr[i,:,j]).nonzero()]
            idx_True = bool_idx_arr[i,:,j].nonzero()
            interpolated = np.interp(idx_True, xp, fp)
            work_dataset[i,:,j][idx_True] = interpolated
    
    return work_dataset