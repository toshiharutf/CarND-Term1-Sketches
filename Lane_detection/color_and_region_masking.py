# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:22:57 2017

@author: Toshiharu
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read int the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ',type(image),
		'with dimensions: ', image.shape)
		
# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
# Note: always make copy rather than simply using "="
color_select = np.copy(image)
line_image = np.copy(image)

# Define our color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest
# Keep in mind the origin (x=0 and y=0) is in the upper left corner

left_bottom = [100, 539]
right_bottom = [820, 539]
apex = [470, 300]

# polyfit syntax: np.polyfit ( (x_Coordinates), (Y_coordinates, Number of poly to fit))
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds =  (image[:,:,0] < rgb_threshold[0]) \
        		    | (image[:,:,1] < rgb_threshold[1]) \
			        | (image[:,:,2] < rgb_threshold[2])
            
## Find the region inside lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
                    
                    
print (color_thresholds.shape)
print (region_thresholds.shape)
                    

# Mask color selection
color_select[color_thresholds | ~region_thresholds] = [0,0,0]

# Find where image is both below the color threshold and inside the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]





# Display our two output image
#plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
print(x)
plt.plot(x, y, 'b--', lw=4)
plt.imshow(color_select)
plt.imshow(line_image)
plt.figure()
plt.imshow(image)