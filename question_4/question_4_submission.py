import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from scipy.interpolate import griddata
import cv2
from scipy.signal import correlate
from scipy.ndimage import shift
import math

## Import Images #############################################################

all_images = np.zeros((1,100,100))

for fits_files in glob.glob('*.fits'):
   
    hdulist = fits.open(fits_files)
    data = np.expand_dims(hdulist[0].data,0)
    all_images = np.concatenate((all_images,data))
    # could do append to an empty array instead
    
all_images = np.delete(all_images,0,0)

## Plots #####################################################################

plt.figure(1)
plt.imshow(all_images[0], cmap = 'gray', vmin = np.min(all_images[0]), vmax = np.max(all_images[0]))
plt.title('Example Image')

## Bad Pixel Locations #######################################################

find_bad_pix = np.log(np.sum(all_images.copy(),0))
plt.figure(3, figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(find_bad_pix)
plt.title('Log of All Images')

badpixel = np.loadtxt('badpixel.mask').astype(int) # load bad pixels
badpixel[:,0] -= 1 # x-coordinates to array indicies
badpixel[:,1] -= 1 # y-coordinates to array indicies

fault = np.zeros((100,100)) 
fault[badpixel[:,1],badpixel[:,0]] = 1 # faulty areas on ALL images
plt.subplot(1,2,2)
plt.imshow(fault) # showing all bad pixels for clarity
plt.title('Bad Pixel Locations')

# Interpolation ##############################################################

def bad_pixel_cosmic(array,badpixel,n):
    
    outliers = abs(array - np.mean(array)) < (n*np.std(array)) # outlier values (cosmic neutrinos)
    array *= outliers
    array[badpixel[:,1],badpixel[:,0]] = 0 # bad pixel locations
    
    #print(len(array[outliers == False]))
    
    array = np.ma.masked_equal(array,0)
    
    x = np.arange(array.shape[1])
    y = np.arange(array.shape[0])
    x_grid, y_grid = np.meshgrid(x, y)
    x_points = x_grid[~array.mask]
    y_points = y_grid[~array.mask]

    array = array[~array.mask]
    interpolation = griddata((x_points, y_points), array,(x_grid, y_grid), method='cubic')
    
    return interpolation

all_images_interp = np.zeros((25,100,100))

for i in range(25):
    new = bad_pixel_cosmic(all_images[i].copy(),badpixel,7)
    all_images_interp[i] = new

plt.figure(4)
plt.imshow(all_images_interp[0], cmap = 'gray')
plt.title('Example Interpolated Image')
    
## Sky Subtraction ###########################################################

def sky(images,n):
    
    means = np.mean(np.mean(images,1),1)
    means = np.reshape(means,(25,1,1))
    images_mean = images/means
    
    for i in range(100):
        
        sliced = images_mean[:,i,:].copy()
        std = np.std(sliced)
        sky = sliced < 1+(n*std)
        images[:,i,:] *= sky 
    
    return images

images_sky = sky(all_images_interp.copy(),2.4) # only sky values
images_sky_mean = np.ma.masked_equal(images_sky,0) # mask to set 0 values to empty
images_sky_mean = np.mean(np.mean(images_sky_mean,1),1) # mean of each images sky
images_sky_mean = np.reshape(images_sky_mean,(25,1,1)) # reshape for division

all_images_interp -= images_sky_mean # remove mean sky value to associated image

plt.figure(5)
plt.imshow(images_sky[0], cmap = 'gray')
plt.title('Example Sky Mask')

## Offset Determination - Method 1 ###########################################

def offset_v1(images,length):
    
    full = np.zeros((25,300,300))
    base = images[0].copy()
    full[0,100:200,100:200] = base.copy()
    
    base -= np.mean(base)
    
    for i in range(24):
        
        compare = images[i+1].copy()
        compare -= np.mean(compare)
        
        corr = correlate(base,compare,mode='full')
        corr_max = np.max(corr)
        ind = np.where(corr == corr_max)
        
        d_x = ind[1]+1-length 
        d_y = ind[0]+1-length
        
        full[i+1,int(100+d_y):int(200+d_y),int(100+d_x):int(200+d_x)] = images[i+1].copy() 
        
    return full

offset_images = offset_v1(all_images_interp.copy(),100)
offset_images = np.ma.masked_equal(offset_images,0) # remove 0 values
composite = np.mean(offset_images.copy(),0) # create composite
final_image_v1 = composite[100:227, 100:227]
plt.figure(6)
plt.imshow(final_image_v1, cmap = 'gray')
plt.title('Final Image - Offset Determination v1')

## Offset Determination - Method 2 ###########################################

scale = 10 # scale to increase images
interp_size = np.zeros((25,100*scale,100*scale)) # grid for enlarged images

for i in range(25):
    image = all_images_interp[i].copy()
    new_image = cv2.resize(image,dsize=(100*scale,100*scale),interpolation=cv2.INTER_CUBIC)
    interp_size[i] = new_image
    
def offset_v2(images,length):
    
    base = images[0].copy()
    base -= np.mean(base)
    delta = np.zeros((24,2))
    
    for i in range(24):
        #print(i)
        compare = images[i+1].copy()
        compare -= np.mean(compare)
        corr = correlate(base,compare)        
        corr_max = np.max(corr)        
        ind = np.where(corr == corr_max)
                
        d_x = ind[1]+1-length 
        d_y = ind[0]+1-length
        
        delta[i,0] = d_x
        delta[i,1] = d_y
        
    return delta

offsets = offset_v2(interp_size, 100*scale)/10 # float offsets

offset_images_v2 = np.zeros((25,300,300))
offset_images_v2[0,100:200,100:200] = all_images_interp[0].copy()

for i in range(24):
    offset_images_v2[i+1,100:200,100:200] = all_images_interp[i+1].copy()
    offset_images_v2[i+1] = shift(offset_images_v2[i+1], [offsets[i,1],offsets[i,0]])
    
    
composite_v2 = offset_images_v2[:,100:200+math.ceil(np.max(offsets[:,1])),100:200+math.ceil(np.max(offsets[:,0]))] # crop
final_image_v2 = np.mean(composite_v2,0) # create composite

plt.figure(7)
plt.imshow(final_image_v2, cmap = 'gray')
plt.title('Final Image - Offset Determination v2')


