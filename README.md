# Imaging and Data Processing Task Sheet

Solutions for task sheet questions from 4th year module Imaging and Data Processing. All questions completed in Python

## Question 1: Simple Image Processing

### Greyscale Enhancement via Point Processing

I first read in my chosen RGB image file, and then created a function which converts RGB to greyscale. This function creates weightings for how much the red, green, and blue colours in the image have an impact on our new image [1].

When displaying this new greyscale image, I had to specify the range over which to plot it as an image (0 to 255) as python naturally scales data from the minimum to the lowest value in the image. This is essentially doing linear scaling automatically.

I then scaled my image between 0 and 1. This is because logarithmic scaling needs to be used for this specific image. Creating a histogram to illustrate the greyscale levels allows you to judge if you should use logarithmic or linear scaling. If the grey levels are distributed such that there are some pixels near the 255-maximum value, then linear scaling would not work as well as logarithmic.

The equation for linear scaling allows all data points to be scaled depending on their current value. For example, if you had a value at 255, it would first be re-scaled to 1 (as 255 is the maximum value). If you then subbed it into the logarithmic scaling equation, the numerator and denominator would be equal and therefore cancel out, and the new value would still be 1. If you have another value at 100, this would be scaled to 0.4, which when subbed into the scaling equation (with a value of alpha = 5) would return 0.6. The lower valued greyscale pixels are increased depending on how low they already are. The higher the value of alpha used, the brighter the image becomes overall. You want to find a good balance between making the image clearer, but not making every element so bright the image looks bad. This works very well for my chosen image, as the greyscale levels are distributed up to the 255 level.

![Greyscale](/Images/greyscale_enhancement.png)

### Image Thresholding

I chose an image with a lot of large, block colours so that the thresholding is more obvious. For example, the image contains the sea, sky, trees, and a white building. 
I started by slicing out the RGB elements of my image into variables. This makes it easier to manipulate them later. I then created variables for the upper and lower threshold limits for all 3 colour channels. I then found all areas in my image where the RGB values all fall between the upper and lower thresholds set each colour channel. I then set the whole image to a matrix of 0's and set all the coordinates in the image that fit the previous threshold criteria to 255. This is simply separating the chosen colour range as white, and everything else as black.

Initially, I did not set the image matrix elements all to 0 before setting the values which met the threshold criteria to 255, and instead just set all values not at 255 to 0. This meant there may be pixels that were already at 255 in the original image. This would result in my image displaying the wanted thresholds, but also a few pixels in only one colour band which were already at 255.

I also didn't initially have the ‘and’ term in finding these pixel coordinates. This caused an issue as there may be an area in which one colour channel threshold criteria is met at a specific pixel, but the same pixel in the other two channels does not meet the threshold criteria. This would mean that the colours I wanted to be thresholded are, but then there are also other patches of individual colours that met their own criteria.

For an example of thresholding, to separate most of the white building in the image, set the lower threshold for all 3 channels to about 190. The threshold values currently in the code should pick out the ocean.

![Segmentation](/Images/segmentation.png)

### Edge Detection

I started by defined my vertical and horizontal Prewitt filter matrices. There are also different filters you could use that look for edges, like a Sobel filter, that are similar to Prewitt filters. I tried edge detection with Sobel filters, but there was not much difference in the end result.

I next defined all dimensions of my image to be used later, as well as an empty array for the final image. The dimensions of this empty array are due to the filtering process decreasing the size of the image. Because the filter is a 3x3 matrix moving through the whole image a step at a time, the image it creates from an image of dimensions [h,w,d] will be [h-2,w-2,d].

I started with a loop over all colour channels, the width of the new image, and the height of the new image. At each step, a 3x3 ‘region’ is created, element-wise multiplication is performed with it and the filter, and the sum of these are then taken. This is done for the vertical and horizontal filter.
To combine the horizontal and vertical edges, both are squared, summed together, and then the root of this is found. This whole process is repeated from edge to edge vertically and horizontally, eventually spanning the whole image to create the edge detected image. The edge detection images for all 3 colour channels are all combined and normalised between 0 and 1 [2].

The property of having an image slightly smaller than the original can be fixed by including padding. This is just adding zeros around the perimeter of the original image, so that the edge detection image is the same size as the original one [2].

![edge1](/Images/edge_detection.png)

Other more efficient techniques can be used to detect the edges of an image, such as canny edge detection. This works similar to my code, but with a few differences. A Gaussian filter is applied to the image beforehand to remove some noise, and the edge filters are applied just like before. Non-maximum suppression and Hysteresis are also applied essentially just removing parts of the image not deemed to be an edge and setting thresholds to change the sensitivity of what is detected to be a true edge. I implemented this using the cv2 library to show another method of edge detection.

I picked my image specifically as it has a lot of obvious corner and edges, as well as text, that will be picked up well to display how the edge detection works. 

(In order to plot the Canny Edge image, the module cv2 is needed)

![edge2](/Images/edge_detection_cv2.png)

## Question 2: Affine Transformations


The coordinates supplied are all to undergo a set of affine transformations at each time step (7200 in total). I started by defining the translational and rotational affine transformations as functions. For example, for the rotational affine transformation I first defined what the affine transformation matrix would be, and put in variables for change in x, y, z. I combined the translational matrix for x, y, z into one matrix as it does not matter what order you apply them in. The output of the function is just this matrix multiplied by the initial coordinates. In order to see how this matrix works, you could put any specific values in and just expand it normally. This would lead to a set of equations describing the change in x, y, and z. Having the transformation in matrix form simply makes everything easier and more compact [4].

For the rotational affine transformations, the same method was used but just with the rotational transformation matrix for rotations about the x, y, and z axes. I did not combine all of these transformations as the order of operation does make a difference to the overall coordinates. I found out later on that while it did change the coordinate values depending on the order you apply them; it was so minimal it made no effect to the animation. Since inputs to my affine functions need to be in the form (x,y,z,1), I reshaped the initial coordinates of the markers and added a matrix of ones to get them in this format [4].

When it comes to finding the markers positions, I had to consider that all of the data given (trans, angles) is from the point of view of the centre of mass (CoM). To calculate the centre of mass, I interpreted this to be the mean position of all markers. So when a rotation is performed for instance, it would make a difference if it were centred at the origin or centred at the CoM. At every time step, I convert all coordinates to be centred at the CoM and sub in these coordinates of the markers into the affine transformations defined earlier. After this step, I converted the markers back to the original coordinate system centred at (0,0,0). 

After this, the new CoM is calculated for the new coordinates, and the next time step occurs. Since the data given (trans, angles) represent the change from the CoM, the affine transformations can be acted on the initial coordinates, and the next set of translational and rotational values can be used. At the end of each time step the new coordinates created are added to the array of zeros defined earlier. Due to how the functions defined output the coordinates, there will be a value of 1 attached to the end of all coordinates that can be sliced out of the array.

When animating, I first created a 3d plot of all the points at t = 0 with the limits of the room. In order to actually animate it, I used the FuncAnimation feature. This takes the initial 3d plot of all points and updates the points plotted according to some function, which is just cycling through the position array. The interval parameter in the animation was set to 8.33, which is automatically in milliseconds since all movement was sampled at 120Hz. This makes the outputted video play at the rate it was recorded. Then the animation is saved to the current directory and is automatically commented out, so the video isn't saved every time the whole program is run. It is worth noting in the animation function part of this, I included a variable 'stepsize', normally set to 1. This is just to speed up the animation to whatever speed [5].

## Question 3: Time-Frequency Decomposition

### Part A

I converted the signal into frequency space using a Fourier Transform. I first calculated what the range of frequencies will be for the transform. Only the first half of the Fourier transform, and associate frequencies, are actually useful. This is because when taking the Fourier transform of a real signal, the output is repeated with symmetry about the central value. Only the positive frequencies are needed. Again while not necessary, plotting the absolute value of Fourier transform against the calculated frequencies will give the power spectrum of all frequencies in the original signal.

The next step involves creating a filter. I decided to use a Butterworth filter for this. I also made a function for this filter to save time when using it many times later on. The inputs for this filter are the order, the critical frequencies, and the btype. The order is, as the name implies, the order of the polynomial that is approximating the filter. For the order, I found that for sectioning out a small range of frequencies, an order of 3 worked well. The critical frequencies are the frequencies you wish to filter between. btype is what type of filter, which is a bandpass in our case, hence why we have two critical frequencies. You could have a high pass or low pass filter instead, which would begin filtering at one specific frequency. The input for critical frequency variables also needs to be normalised between 0 and the Nyquist frequency for this specific filter function, so just half of the sampling rate [6][7].

I tested plotting my filter just for a sanity check. I assigned the outputs from the filter to ‘a’ and ‘b’. These are the denominator and numerator respectively of the polynomial that will approximate the function. When it comes to actually creating a filter though, the function freqz is used, with outputs ‘w’ and ‘h’. It essentially takes your ‘a’ and ‘b’ values and plots them over your desired frequency range. When plotting the filter, it is just the original frequency range against the magnitude of ‘h’. I could have used the ‘w’ variable and plotted (sample rate/(2 x pi)) x w instead of the frequencies calculated before. This is as w is just an array the same length as our Fourier transform that ranges from 0 to pi. Both achieve the same thing. This filter will have a maximum value of 1 centred between the desired frequency range, and then drop to 0 moving outside this range. It's important to not have a discontinuous drop to 0 however, as this will result in ringing when converting the signal back to the time domain. The Butterworth filter is good in this regard as it is easy to create and the decay to 0 is not discontinuous, but gradual, ensuring minimal ringing [7].

Multiplying the filter by the Fourier transform I had from before will basically ‘cut out’ everything not in the desired frequency range. I then took the inverse Fourier transform of this to create the original signal, but only showing the frequencies that have been chosen using the filter. Taking the Hilbert transform on the real parts of this new signal creates an envelope of the amplitude of this signal. So this is just how strong this specific frequency band is over time. Again, I plotted for a sanity check.
The next step was to loop this entire process over a range of frequencies to create a Time-Frequency Distribution (TFD). So for example do this for frequencies 1-3Hz, then 2-4Hz, then 3-5Hz etc. Another aspect that would affect this is how much you increase your first critical frequency by ('stepsize'). For example I started as 1-3Hz, then 2-4Hz etc. Instead you could instead do 1-3Hz, then 1.5-3.5Hz, or 1.1-3.1Hz. This increases the time taken to run but up to a point does make the TFD clearer as you are sampling more. 'endfreq' is up to which frequency you want to plot up to. So for the full range it would just be the maximum value of all frequencies (300). I have set it to 60 to see the more interesting parts of the TFD.

For creating the TFD, I would calculate the envelope of amplitude for my filtered signal at some frequency range, and then add this to an empty array. The envelope is going to be a 1x (length of time) array. So by repeating this process and stacking the envelope amplitudes on top of each other, you are building up how the amplitudes of each frequency range is varying over time.

![tfd](/Images/TFD.png)

### Part B

This uses similar methods to 3a with slight changes. I again defined all useful information for the signal (time, sample rate, time interval), but then I defined the same but for the length of a single task. The task takes 6.5s and is repeated 95 times, and the sample rate is 600Hz. So each task will be 3900 values in the signal brain array. I defined the same Butterworth filter function from 3a as it will be used a lot. The function defined after it however is what will be doing all the work, which looks at 6.5-second-long segments of the original signal to be picked out. With this new segmented signal 6.5 seconds long, the same steps as 3a are used.

The main difference comes when we use this function to create an average TFD. I looped over every task in the original signal. After each iteration in the loop, each TFD is added to the previous. This is why the empty array of zeros created before the loop is important. This new image of 95 TFD added together is then normalised by the number of loops (95 tasks). This isn’t important for plotting as is but will be when showing the percentage change from baseline.

To calculate the baseline, the last second of the signal is used, when the subject is not performing a task. The baseline TFD is calculated using the same function as before, but over the last second. With this TFD for 1 second, an average of the baseline can be found. The original TFD baseline array will be (frequency range x sampling rate) in size. To create an average for what the frequency range strengths should be, I took the mean along the time axis. This mean value for each frequency then needs to stretch to be the same size as our original TFD for the tasks. This creates the same size array where the distribution of strengths of frequencies will be the same at every time, so basically an average of what the frequencies power distribution should be at any point in time. In order to calculate the percentage change I took the original image, divided it by the baseline array, multiplied it by 100, and then subtracted 100. So if a value in the task TFD was 20, and the associate value in the baseline was 10, this would result in a percentage change of 100.

I have also allowed the user to select which signal to view the TFD for by selecting either signal at the start.

![tfd2](/Images/TFD_signal_brain.png)

![tfd3](/Images/TFD_signal_brain_percentage.png)

### Part C

Using a filter to ‘segment’ out specific frequencies and plot the intensity of these frequencies against time is similar in concept to taking the short time Fourier transform. This is when you take the Fourier transform over a very small region of time in the signal. The Fourier transform at each time window will show the strengths of each frequency in the original signal. By plotting this strength of each frequency at lots of short time intervals, you can build up a TFD like before. Instead of plotting at each frequency window (like for Parts A and B), you plot at each time window. The issue with this is you can have either a high frequency resolution or a high time resolution, but not both since you have a constant time interval [8].

An alternative to this is the continuous wavelet transform (CWT). A wavelet is some waveform with an average value of zero of  short time duration. The wavelets properties can be adjusted to stretch/squeeze , as well as displace it by some amount. The CWT works by calculating the similarity (convolution) between a chosen wavelet and the signal. The wavelet is displaced along the signal, until the similarity between all of the signal and the wavelet has been found. This is repeated for a range of scaled wavelets, until all scaled wavelets have been used. When this is done, the similarity will have been found for all shift and scales, essentially creating a 3-dimesnional array. In this case, the scale corresponds to the frequency and the shift corresponds to the time. When plotting this data as a heatmap/scalogram, a TFD has been made. An issue with using CWT is it contains areas that have been affected by edge artifacts due to the wavelet expanding over a region that does not contain the signal (edges).[9]

(scipy.signal and scipy.fft are required to run this code)

## Question 4

### Combination of Multiple Near-Infrared Images

To read an image in, fits from astropy.io is used. fits.open is what opens the file, but to actually save it you need to specify you want the data aspect of the file saved. I used the glob toolkit in python to read all the files as once, assigning them suitable names. I saved them all in one array of size 25,100,100. I displayed an example image with specific limits, but python also does this automatically anyway [10].

![example](/Images/example_image.png)

By adding up all 25 images, taking the log of each pixel, and then plotting as an image, the bad pixel values are able to be seen. They are quite obvious from the figure displayed, which clarifies how to interpret the original bad pixel array.

![pixel](/Images/bad_pixel_locations.png)

For each array, I calculated the mean of all pixels, and looked for values that deviate greatly from the mean. These values will be where a cosmic ray has hit the pixel. Setting how far away from the mean will change how many of these pixels are removed. I set it to an amount to remove the vast majority of the cosmic burst, but not remove the actual sources. If there are one or two leftovers, they will be averaged away later on. 

To actually remove the locations of these rays and bad pixels, all the coordinates of such items are set to 0. A mask is then applied to all values equal to 0, setting the value in those coordinates to be empty. All of the array indices and values of the image that haven't been masked are used as data to interpret data in the locations that have been masked. This is done using the cubic interpolation. At the desired pixel to interpolate, the surrounding neighbourhood of pixels is essentially averaged based on how close it is the desired pixel, and this value is placed in the desired pixel [11][12].

![interp](/Images/example_interpolation.png)

In order to subtract the sky value in each image, each image is divided by its associated mean value. This means that if you plotted the values of some selected pixels as a function of image number, the plot would be flat, with sharp high values if an object is present. If the images were not divided by their mean values, the plot would have some gradient, with higher values if an object is present. The flatter line is easier to work with. All values along this plot that deviate from the mean of the line are noted as not being the sky. Repeating this over the whole 'cube' array of 25 images stacked will allow a mask to be made over all images, the same as done for the bad pixels and cosmic rays. With the array of all images containing only sky data, the mean for each image can then be calculated and subtracted from the relevant image.

In order to determine the offsets of each image relative to each other, I used cross-correlation [13][14]. This is basically seeing how similar one image is compared to another. It is like taking the convolution between your images and having an outputted array of how similar the two images are at different offsets. The offset at which the images are most similar is when this outputted array is a maximum. First, both images are re centred relative to their mean, so just having the mean taken away from each image [15]. The cross-correlation will not have a meaningful output if both inputs do not have a non-zero mean. The maximum value of the convolution and its indices are then found. The information on what the x and y offsets are is contained in these indices. The x and y value have the width and height of the image subtracted respectively to calculate the offset (+1 due to python zero indexing). With this, you now know how far to shift your second image to make it 'fit' over the first image. I repeated this for all images to get their offsets relative to the first image. 

To get an image off all these images, a 25x300x300 zero array was made, with the first image placed in the centre on the first plane. When the offset for an image is found, the image is copied to another plane, shifted by the offset amount relative to the first image. This is repeated for all images. The reason for the 300x300 grid is since I did not know how large the offsets would be. Once all images with offsets have been found, all 0 values are masked since they were not in the original data, and then the mean of all images stacked on top of each other is found. I then cropped the image down to make it easier to see. The final image was 127x127 in size. I did this manually but could easily be done by cropping between the maximum offsets, which I do in the next part. 

![example](/Images/final_image_none.png)

This final image does look good, but it has been done with integer offsets. An issue with this is that it gives a 'best' fit for images. The offsets found would realistically be non-integer values, which the above does not consider. A solution to this is to increase the resolution of each image via interpolation, so for example interpolating from a 100x100 image to a 1000x1000 array. When interpolating the images, the offsets will be larger. These offsets can then be scaled back down to give a non-integer offset.
To do this, I first scaled all my images by some factor (currently 10), which creates a new array of all images of size 1000x1000 [16]. This is another use of cubic interpolation. Just like when removing the bad pixel and cosmic ray pixels, the local neighbourhood (4x4 in this case) is looked at for each pixel to be able to best assign a pixel value. Once this is done, the same method as before for determining offsets is done. These offsets will still be integers, but now they can be scaled down. For example, an offset of 237 in a 1000x1000 image will be 23.7 in the original image.

In order to shift an image by a non-integer amount, interpolation is used again. Like from the examples given in the lecture on interpolation, if you had a centroid of a pixel that doesn’t lie exactly in the centre of a pixel, you look at the surrounding pixels and ‘distribute’ the pixels value among local pixels depending on how far they lie from the centroid. The scipy function shift can do this automatically [17]. If the 100x100 images were shifted using this method however, part of the image would be cut off since the output would still be a 100x100 image. To solve this problem, I created a 25x300x300 array of zeros (like for the first method) and put all images in the centre of each plane. This meant that when it came to shift all images, none of the original image we want is lost. After shifting all images, the whole array can be cropped. The cropping region can be found by taking the original 100x100 section and extending it by the maximum and minimum offsets in the x,y directions. Once cropped, I took the mean value of each slice, creating a 127x127 image. This is the same size as the previous final image.

![example](/Images/final_image_interpolation.png)

(The modules needed to run this code are: scipy, cv2, glob, astopy.io)

## References

[1] https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

[2] https://victorzhou.com/blog/intro-to-cnns-part-1/

[3] https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html

[4] https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf

[5] https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python

[6] https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7

[7] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

[8] Image and Image Processing – Student Videos

[9] Image and Image Processing – Student Videos – CWT

[10] https://docs.astropy.org/en/stable/io/fits/

[11] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

[12] https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python/39596856#39596856

[13] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html

[14] https://uk.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html?fbclid=IwAR3KBgZt3YvYmbt93ikcF3bZ1ri9SoLfoBfqv-8moAKbxwAOQRIj9hJcGdM

[15] https://math.stackexchange.com/questions/840115/correlation-coefficient-calculation

[16] https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

[17] https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html





