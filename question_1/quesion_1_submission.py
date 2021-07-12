import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

## QUESTION 1i) ##############################################################

img_coloured = mpimg.imread('enhance.jpg')

def greyscale(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    grey = (0.299*r) + (0.587*g) + (0.114*b)
    return grey

grey = greyscale(img_coloured)

plt.figure(0, figsize=(15,6))
plt.subplot(1,3,1)
plt.imshow(grey, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('Original greyscale image')

# plot of grey distribution for greyscale image
# flatten does as expected, flattens to 1D array AND creates copy (not like ravel)

grey = grey/(np.max(grey))
alpha = 10
grey_new = (np.log(1+(alpha*grey))) / (np.log(1+alpha))
# normalise original RGB image to grey

plt.subplot(1,3,2)
plt.hist(grey.flatten(), bins = 256)
plt.hist(np.ravel(grey_new), bins = 256)
plt.ylabel('N')
plt.xlabel('Grey level')
plt.legend(['Normal', 'Enhanced'], loc='best')

plt.subplot(1,3,3)
plt.imshow(grey_new, cmap = 'gray',vmin = 0, vmax = 1)
plt.title('Enhanced greyscale image')
# non-linear scaling of image

## QUESTION 1ii) #############################################################

image = mpimg.imread('threshold.jpg').astype('uint16')
plt.figure(1, figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title('Original Image') 

R = image[:,:,0] # red channel
G = image[:,:,1] # green channel
B = image[:,:,2] # blue channel

h_r = 65 # high threshold (red)
l_r = -1 # low threshold (red)
h_g = 110 # high threshold (green)
l_g = 15 # low threshold (green)
h_b = 190 # high threshold (blue)
l_b = 100 # low threshold (blue)

true_coords = np.where((R>l_r) & (G>l_g) & (B>l_b) & (R<h_r) & (G<h_g) & (B<h_b))
image *= 0

image[:,:,0][true_coords[0],true_coords[1]] = 255
image[:,:,1][true_coords[0],true_coords[1]] = 255
image[:,:,2][true_coords[0],true_coords[1]] = 255

plt.subplot(1,2,2)
plt.imshow(image)
plt.title('Segemented Image')

## QUESTION 1iii) ############################################################

img = mpimg.imread('edge.jpg').astype('uint16') 
img = img/np.max(img) # normalise between 0 and 1 
plt.figure(2, figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')

filter_v = np.array(([-1,0,1],[-1,0,1],[-1,0,1])) 
# vertical filter (Prewitt)
filter_h = np.array(([-1,-1,-1],[0,0,0],[1,1,1])) 
# horizontal filter (Prewitt)

h, w, d = img.shape
# height, width, depth

img_edge = np.zeros([h-2,w-2,d])
# empty array for final image. Filter causes smaller dimensions

for k in range(d):  
    for i in range(h-2):
        for j in range(w-2):
            
            region = img[i:i+3, j:j+3, k] # creates 'box' for our filter
            edges_v = np.sum(filter_v * region)
            edges_h = np.sum(filter_h * region)
            
            edge_tot = np.sqrt(edges_v**2 + edges_h**2)
            img_edge[i,j,k] = edge_tot
            
img_edge = np.sum(img_edge,2)
img_edge = img_edge/np.max(img_edge)

plt.subplot(1,2,2)
plt.imshow(img_edge, cmap = 'gray')
plt.title('Edge Detected Image (Kernel)')

## cv2 Method ###############################################################

import cv2

img = mpimg.imread('edge.jpg')
edges_cv2 = cv2.Canny(img, 40, 200)
plt.figure()
plt.imshow(edges_cv2, cmap = 'gray')
plt.title('Canny Edge of Image')

