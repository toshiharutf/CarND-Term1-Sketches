import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read int the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ',type(image),
		'with dimensions: ', image.shape)
		
# Grab the x and y size and make a copy of the image
xsize = image.shape[0]
ysize = image.shape[1]
# Note: always make copy rather than simply using "="
color_select = np.copy(image)

# Define our color selection criteria
# Note: if you run this code, you'll find these are not sensible values!

red_threshold = 240
green_threshold = 240
blue_threshold = 240
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

#Identify pixels below the threshold
thresholds =  (image[:,:,0] < rgb_threshold[0]) \
			| (image[:,:,1] < rgb_threshold[1]) \
			| (image[:,:,2] < rgb_threshold[2])
			
color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)
plt.show()
