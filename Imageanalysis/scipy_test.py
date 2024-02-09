from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

__loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
filename = 'c.97_500X-y8000_p3.JPG'
filepath = os.path.join(__loc__,filename)

f = plt.imread(filepath, format='jpg')
f = Image.open(filepath).convert('L')
print(f)
arr = np.asarray(f)
grad = arr[:,:] - arr[:,:]
print(arr.shape)
#f = np.fromfile(filepath)
#f = misc.face()
#misc.imsave('c.97_500X-y8000_p3.JPG', f) # uses the Image module (PIL)



plt.imshow(arr, cmap='inferno', vmin=0, vmax=255)
#plt.imshow(f,cmap='Greys_r')
plt.show()