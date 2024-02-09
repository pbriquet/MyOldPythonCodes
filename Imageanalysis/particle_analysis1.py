import cv2
import pylab
from scipy import ndimage
import os

__loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
filename = 'c.97_500X-y8000_p3.JPG'
filepath = os.path.join(__loc__,filename)

im = cv2.imread(filepath)
pylab.figure(0)
pylab.imshow(im)

print(im.shape)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
maxValue = 255
adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C#cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
blockSize = 5 #odd number like 3,5,7,9,11
C = -3 # constant to be subtracted
im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C) 
labelarray, particle_count = ndimage.measurements.label(im_thresholded)
print(particle_count)
pylab.figure(1)
pylab.imshow(im_thresholded)
pylab.show()