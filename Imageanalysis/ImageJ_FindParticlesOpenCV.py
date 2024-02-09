import cv2
import pylab
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import numpy as np
import pandas as pd
import os

if __name__=='__main__':
    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    filename = 'c.97_500X-y8000_p3.JPG'
    filepath = os.path.join(__loc__,filename)

    im = cv2.imread(filepath)

    img2 = np.pad(im.copy(), ((1,1), (1,1), (0,0)), 'edge')
    # call openCV with img2, it will set all the border pixels in our new pad with 0
    # now get rid of our border
    im = img2[1:-1,1:-1,:]
    # img is now set to the original dimensions, and the contours can be at the edge of the image

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    columns = ['Perimeter','Area','Contours']
    conts = [cnt for cnt in contours]
    perimeters = [cv2.arcLength(cnt,True) for cnt in contours] # Get the Perimeters of each Contour. True is for closed contours.
    areas = [cv2.contourArea(cnt) for cnt in contours]
    hulls = [cv2.convexHull(cnt) for cnt in contours]

    df = pd.DataFrame(list(zip(perimeters, areas,conts)), columns=columns)
    df = df.sort_values(by=['Perimeter','Area'],ascending=False)

    print(df.head(10))
    index = df.iloc[3].name

    plt_image = cv2.drawContours(im, contours, -1, (0,255,0), 3)
    #plt_image = cv2.drawContours(im, [contours[index]], -1, (0,255,0), 3)
    plt.imshow(plt_image)
    plt.show()
    #cv2.imshow("Keypoints", im)
    #cv2.waitKey(0)
