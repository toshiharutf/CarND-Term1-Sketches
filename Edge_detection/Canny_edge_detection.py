# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:49:36 2017

@author: Toshiharu
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
imshape = image.shape
#vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
vertices = np.array([[(0,imshape[0]),(450, 250), (500,250), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
#poly_X = np.take(vertices,0,axis=2)
#poly_Y = np.take(vertices,1,axis=2)
#print(vertices)
#print(poly_X)
#print(poly_Y)
# Show original picture with the contour of the mask.
plt.imshow(gray)
plt.contour(mask,colors='b',linestyles='dashed')

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 90  #1   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30 #5 #minimum number of pixels making up a line
max_line_gap = 5 #1   # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
#print(lines.shape)

#print(lines[5])

# Iterate over the output "lines" and draw lines on a blank image
#for line in lines:
#    for x1,y1,x2,y2 in line:
#        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10) #draw lines over Hough transformed image

#leftPoints = np.zeros(shape = (lines.shape[0],2) )
#rightPoints = np.zeros(shape = (lines.shape[0],2) )
leftPoints  =[[],[]]
rightPoints =[[],[]]

for line in lines:
    for x1,y1,x2,y2 in line:
        #if m< 0, left line
        if ((y1-y2)/(x1-x2))<0:
            leftPoints=np.append(leftPoints,[[x1,x2],[y1,y2]],axis=1)
        else:
           rightPoints=np.append(rightPoints,[[x1,x2],[y1,y2]],axis=1)
           
print(lines)  
print(leftPoints)              
print(leftPoints[0,:])
print(rightPoints)

leftLine=np.polyfit(leftPoints[1,:],leftPoints[0,:],1)  #x=f(y)
rightLine=np.polyfit(rightPoints[1,:],rightPoints[0,:],1)  #x=f(y)

# Left line initial and end points
y1_left = imshape[0]
x1_left = int(y1_left*leftLine[0]+leftLine[1])

y2_left = 300
x2_left = int(y2_left*leftLine[0]+leftLine[1])

cv2.line(line_image,(x1_left,y1_left),(x2_left,y2_left),(255,0,0),10)

# Right line initial and end points
y1_right = imshape[0]
x1_right = int(y1_right*rightLine[0]+rightLine[1])

y2_right = 300
x2_right = int(y2_right*rightLine[0]+rightLine[1])

cv2.line(line_image,(x1_right,y1_right),(x2_right,y2_right),(255,0,0),10)
        
        #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10) #draw lines over Hough transformed image

# Create a "color" binary image to combine with line image
#color_edges = np.dstack((edges, edges, edges)) 
#
## Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.figure()
plt.imshow(lines_edges)

