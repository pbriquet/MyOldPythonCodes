import cv2
import pylab
from scipy import ndimage
from PIL import Image
import numpy as np
import os

__loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
filename = 'c.97_500X-y8000_p3.JPG'
filepath = os.path.join(__loc__,filename)
image = Image.open(filepath)


blur_parameter = 0.8
original_grayscale = np.asarray(image.convert('L'), dtype=float)
blurred_grayscale = ndimage.filters.gaussian_filter(original_grayscale,blur_parameter)
difference_image = original_grayscale - (multiplier * blurred_grayscale)
image_to_be_labeled = ((difference_image > threshold) * 255).astype('uint8')  # not sure if it is necessary

labelarray, particle_count = scipy.ndimage.measurements.label(image_to_be_labeled)

print(particle_count)
pylab.figure(1)
pylab.imshow(image_to_be_labeled)
pylab.show()

