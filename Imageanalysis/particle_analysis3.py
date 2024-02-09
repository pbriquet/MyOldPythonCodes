import numpy as np
import scipy
import pylab
import pymorph
import mahotas
from scipy import ndimage
import cv


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = ndimage.morphology.generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = ndimage.filters.maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks


__loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
filename = 'c.97_500X-y8000_p3.JPG'
filepath = os.path.join(__loc__,filename)
im = mahotas.imread(filepath)
imf = ndimage.gaussian_filter(im, 3)
#rmax = pymorph.regmax(imf)
detected_peaks = detect_peaks(imf)
pylab.imshow(pymorph.overlay(im, detected_peaks))
pylab.show()