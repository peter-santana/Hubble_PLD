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
	testing_array = np.zeros(hdul[0].data.shape)
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

def orbit_excise(STIS_dataset, orbit_to_excise = 1):
	"""
	"""
	return ??

def scan_excise(STIS_dataset, scans_to_excise = ['example', 'items']):
	"""
	"""
	return ???

def STIS_select_aperture(STIS_dataset, center_pixel = 57, aperture_width = 10):
	"""
	Function for reducing a STIS_dataset to a selected aperture.

	Parameters
	----------
	center_pixel : integer
		Selects the row (indexed from top to bottom in the spatial dimension)
		that will center the aperture.
	aperture_width : integer
		Selects the distance from the center_pixel that the aperture will span
		in each direction.

	Returns
	-------
	STIS_dataset : 3D numpy array
		Returns a numpy array with a time dimension [explain better]
	"""
	top_border = center_pixel - aperture_width
	bottom_border = center_pixel + aperture_width
	sub_array = STIS_dataset[:,top_border:bottom_border,:]
	return sub_array

	# WORK IN PROGRESS!!!!!

def get_mad_sigma(x):
    """
    Estimate the robust version of the standard deviation (sigma) using the Median Absolute Deviation (MAD).
    See: https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(x-np.median(x)))*1.4826

def standarize(x):
    return (x-np.median(x))/get_mad_sigma(x)

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

def pixelify(STIS_dataset, Npixels = 16):
	"""
	Function to cut up a STIS_dataset into discrete white-light pixels.

	Parameters
	----------
	STIS_dataset : 3D numpy array

	Npixels : integer
		Number of individual pixels that the dataset will be divided into.
	"""
	Ntimes = len(STIS_dataset)
	Pixels_Array = np.zeros((Ntimes, Npixels),dtype= 'f8')
	for i in range(Ntimes):
	    new_data = STIS_dataset[i]
	    sub_arrays = np.array_split(new_data, Npixels, axis = 0)
	    pixel_list = []
	    for j in range(Npixels):
	        Pixel = sub_arrays[j]
	        Flux = np.sum(Pixel)
	        Pixels_Array[i,j] = Flux

	Array = np.hstack((date_array, Pixels_Array)) #keep this???

	return pixel_dataset

def normalize_pixels(pixel_dataset):
	"""
	"""
	return ???

def get_phases(t,P,t0):
    t = t + 2400000.5
    """ 
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:   
        phase = ((t - np.median(t0))/np.median(P)) % 1
        if phase>=0.5:
            phase = phase - 1.0
    return phase


def regressed_PLD(normalized_pixels, out_of_transit_indices):
	"""
	"""

	# Regress PLD on those out-of-transit datapoints. Add a matrix of ones to account for the A coefficient:
	X = np.vstack(( np.ones(Phat.shape[0]), Phat.T ))
	# Perform linear regression to obtain coefficients on out-of-transit data:
	result = np.linalg.lstsq(X.T[idx_oot,:], f[idx_oot], rcond=None)
	coeffs = result[0]

	return ???

def standarize_data(input_data):
    """
    Standarize the dataset
    """
    output_data = np.copy(input_data)
    averages = np.median(input_data,axis=1)
    for i in range(len(averages)):
        sigma = get_mad_sigma(output_data[i,:])
        output_data[i,:] = output_data[i,:] - averages[i]
        output_data[i,:] = output_data[i,:]/sigma
    return output_data

def classic_PCA(Input_Data, standarize = True):
    """  
    classic_PCA function
    Description
    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols,eigenvalues,eigenvectors_rows = np.linalg.svd(np.cov(Data))
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:,idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1],:]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows,eigenvalues,np.dot(eigenvectors_rows,Data)







