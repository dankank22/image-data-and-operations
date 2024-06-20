# Lab 3 Report

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image 
%matplotlib inline

from IPython.display import Image as ipyimage #For displaying images in colab jupyter cell

### Exercise 1: Generalized function for subsetting pixels

ipyimage('lab3_exercise1.PNG', width = 1000)

e1_img = Image.open('sample_image_1.jpg').convert('L')
e1_img = np.array(e1_img)     # 8-bit code grayscale

fig = plt.figure(figsize=(7, 7))

plt.imshow(e1_img, cmap = 'gray')

def subset_pixels(image, min_pixel_depth, max_pixel_depth, replacement_val):
     
  img_gray_copy = image.copy()            # Copy the image to a new variable to avoid operating on a same image
  
  img_gray_copy[img_gray_copy < min_pixel_depth] = replacement_val    #Apply boolean mask
  img_gray_copy[img_gray_copy > max_pixel_depth] = replacement_val
    
  return img_gray_copy                    # Return the new image

# min_pixel_depth = 50, max_pixel_depth = 250, replacement_val = 100

fig = plt.figure(figsize=(7, 7))

e1_output1 = subset_pixels(image = e1_img, min_pixel_depth = 50, max_pixel_depth = 250, replacement_val = 100)

plt.imshow(e1_output1, cmap = 'gray', vmin = 0, vmax = 255)

plt.savefig('e1_output1.png')

# min_pixel_depth = 100, max_pixel_depth = 200, replacement_val = 30

fig = plt.figure(figsize=(7, 7))

e1_output2 = subset_pixels(image = e1_img, min_pixel_depth = 100, max_pixel_depth = 200, replacement_val = 30)

plt.imshow(e1_output2, cmap = 'gray', vmin = 0, vmax = 255)

plt.savefig('e1_output2.png')

# min_pixel_depth = 50, max_pixel_depth = 255, replacement_val = 255

fig = plt.figure(figsize=(7, 7))

e1_output3 = subset_pixels(image = e1_img, min_pixel_depth = 50, max_pixel_depth = 255, replacement_val = 255)

plt.imshow(e1_output3, cmap = 'gray', vmin = 0, vmax = 255)

plt.savefig('e1_output3.png')

### Exercise 2: Thumbnail generator function

ipyimage('lab3_exercise2.PNG', width = 1000)

e2_img = mpimg.imread('sample_image_2.jpg')
e2_img = np.array(e2_img)     

fig = plt.figure(figsize=(7, 7))

plt.imshow(e2_img)


    
def create_thumbnail(image, downsampling_rate):
    
    height, width, channel = image.shape
    height_copy = height // downsampling_rate
    width_copy = width // downsampling_rate

    output_img = np.zeros((height_copy, width_copy, 3), dtype=np.uint8) # converts to 8 bit (0-255)

    # i and j iterates through the image based on the downsampling rate
    for i in range(height_copy):
        for j in range(width_copy):
            i_0 = i * downsampling_rate
            j_0 = j * downsampling_rate
            i_x = i_0 + downsampling_rate
            j_x = j_0 + downsampling_rate
    # segment finds the 2d array within the given start and end i and j values
            segment = image[i_0:i_x, j_0:j_x, :]
            avg_color = np.mean(segment, axis=(0, 1))
            output_img[i, j, :] = avg_color.astype(np.uint8)
    
    return output_img



# downsampling rate x5

fig = plt.figure(figsize=(7, 7))

e2_output1 = create_thumbnail(image = e2_img, downsampling_rate = 5)

plt.imshow(e2_output1, vmin = 0, vmax = 255)

plt.savefig('e2_output1.png')

# downsampling rate x10

fig = plt.figure(figsize=(7, 7))

e2_output2 = create_thumbnail(image = e2_img, downsampling_rate = 10)

plt.imshow(e2_output2, vmin = 0, vmax = 255)

plt.savefig('e2_output2.png')

# downsampling rate x20

fig = plt.figure(figsize=(7, 7))

e2_output3 = create_thumbnail(image = e2_img, downsampling_rate = 20)

plt.imshow(e2_output3, vmin = 0, vmax = 255)

plt.savefig('e2_output3.png')

## Extra credit: Code efficiency
### Achieve a runtime speed of < 100ms

timeit -n 1 -r 7 e2_output2 = create_thumbnail(image = e2_img, downsampling_rate = 5)

### Exercise 3: Generalized image blender function

ipyimage('lab3_exercise3.PNG', width = 1000)

e3_img1 = mpimg.imread('sample_image_1.jpg')
e3_img2 = mpimg.imread('sample_image_2.jpg')
e3_img3 = mpimg.imread('sample_image_3.jpg')
e3_img4 = mpimg.imread('sample_image_4.jpg')
e3_img5 = mpimg.imread('sample_image_5.jpg')

e3_img1 = np.array(e3_img1)     
e3_img2 = np.array(e3_img2)     
e3_img3 = np.array(e3_img3)     
e3_img4 = np.array(e3_img4)     
e3_img5 = np.array(e3_img5)     

fig = plt.figure(figsize=(15, 15))

plt.subplot(1,5,1)
plt.imshow(e3_img1, vmin = 0, vmax = 255)

plt.subplot(1,5,2)
plt.imshow(e3_img2, vmin = 0, vmax = 255)

plt.subplot(1,5,3)
plt.imshow(e3_img3, vmin = 0, vmax = 255)

plt.subplot(1,5,4)
plt.imshow(e3_img4, vmin = 0, vmax = 255)

plt.subplot(1,5,5)
plt.imshow(e3_img5, vmin = 0, vmax = 255)

def blend_images(image_list, weight_list):

    #initialise variables to store blended values for each colour channel
    
    blended_red = 0
    blended_blue = 0
    blended_green = 0
    
    # iterate through each image and multiply the pixels by the respective weight and added to the blended result
   
    for index, image in enumerate(image_list):
        blended_red += image[:,:,0]*weight_list[index] 
        blended_blue += image[:,:,2]*weight_list[index] 
        blended_green += image[:,:,1]*weight_list[index] 
    
    blended_img = np.stack([blended_red, blended_green, blended_blue], axis = 2)
    blended_img = blended_img.astype(int) # multipolication with the float weights can create float values in our 
    
    return blended_img

# Blend all 5 images with equal weights

e3_part1_image_list =[e3_img1, e3_img2, e3_img3, e3_img4, e3_img5]
e3_part1_weight_list = [0.2, 0.2, 0.2, 0.2, 0.2]

e3_output1 = blend_images(image_list = e3_part1_image_list, weight_list = e3_part1_weight_list)

fig = plt.figure(figsize=(7, 7))

plt.imshow(e3_output1, vmin = 0, vmax = 255)

plt.savefig('e3_output1.png')

# Blend first 3 images with different weights

e3_part2_image_list =[e3_img1, e3_img2, e3_img3]
e3_part2_weight_list = [0.2, 0.3, 0.5]

e3_output2 = blend_images(image_list = e3_part2_image_list, weight_list = e3_part2_weight_list)

fig = plt.figure(figsize=(5, 5))

plt.imshow(e3_output2, vmin = 0, vmax = 255)

plt.savefig('e3_output2.png')

### Exercise 4: Image rotation function

ipyimage('lab3_exercise4.PNG', width = 1000)

e4_img = mpimg.imread('sample_image_2.jpg')
e4_img = np.array(e4_img)     

fig = plt.figure(figsize=(7, 7))

plt.imshow(e4_img, vmin = 0, vmax = 255)

def rotate_image(image, rotate_angle):
 
    img_copy = image.copy()
    
    if rotate_angle == 90:   #rotate by 90 degrees
        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=1) # axis 1 flips horizontally
        return img_copy

    elif rotate_angle == 270:    # rotate by 270 degrees
        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=0) # axis 0 flips vertically
        return img_copy

    elif rotate_angle == 180:    #rotate by 180 degrees
        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=1)
        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=1)
        return img_copy

    else:                        #rotate by 0 degrees
        return img_copy

# Rotate the image by 0 degrees - This should result in identity

e4_output1 = rotate_image(image = e4_img, rotate_angle = 0)

fig = plt.figure(figsize=(7, 7))

plt.imshow(e4_output1, vmin = 0, vmax = 255)

plt.savefig('e4_output1.png')

# Rotate the image by 90 degrees

e4_output2 = rotate_image(image = e4_img, rotate_angle = 90)

fig = plt.figure(figsize=(7, 7))

plt.imshow(e4_output2, vmin = 0, vmax = 255)

plt.savefig('e4_output2.png')

# Rotate the image by 180 degrees

e4_output3 = rotate_image(image = e4_img, rotate_angle = 180)

fig = plt.figure(figsize=(7, 7))

plt.imshow(e4_output3, vmin = 0, vmax = 255)

plt.savefig('e4_output3.png')

# Rotate the image by 270 degrees

e4_output4 = rotate_image(image = e4_img, rotate_angle = 270)

fig = plt.figure(figsize=(7, 7))

plt.imshow(e4_output4, vmin = 0, vmax = 255)

plt.savefig('e4_output4.png')

## Extra credit: Code efficiency
### Achieve a runtime speed of < 5ms

timeit -n 1 -r 7 e4_output2 = rotate_image(image = e4_img, rotate_angle = 90)

### Exercise 5: 2D Gaussian image generator

ipyimage('lab3_exercise5.PNG', width = 1000)

x_range = np.arange(-25, 25, 1)
y_range = np.arange(-25, 25, 1)[::-1]

X, Y = np.meshgrid(x_range, y_range) # More detail on documentation 
                                     # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

print(X, X.shape) # Set of x-coordinates in the function domain     

print(Y, Y.shape) # Set of y-coordinates in the function domain   

def vis_2d_gaussian(X, Y, sigma_x, sigma_y, x0, y0, A, cmap):

    Z = A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2)))
    
    # Create figure and axis for the plot
    figure, axis = plt.subplots(figsize=(7, 7))
    
    # Display the 2D Gaussian
    cax = axis.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)
    figure.colorbar(cax)
   



# First parameter set: sigma_x = 10, sigma_y = 10, x0 = 0, y0 = 0
# cmap = 'gray'

fig = plt.figure(figsize=(7, 7))

vis_2d_gaussian(X, Y, sigma_x = 10, sigma_y = 10, x0 = 0, y0 = 0, A = 255, cmap = 'gray')

plt.savefig('e5_output1.png')

# First parameter set
# cmap = 'jet'

fig = plt.figure(figsize=(7, 7))

vis_2d_gaussian(X, Y, sigma_x = 10, sigma_y = 10, x0 = 0, y0 = 0, A = 255, cmap = 'jet')

plt.savefig('e5_output2.png')

# Second parameter set: sigma_x = 5, sigma_y = 5, x0 = 3, y = 3
# grayscale color specturm

fig = plt.figure(figsize=(7, 7))

vis_2d_gaussian(X, Y, sigma_x = 5, sigma_y = 5, x0 = 3, y0 = 3, A = 255, cmap = 'gray')

plt.savefig('e5_output1.png')

# Second parameter set: sigma_x = 5, sigma_y = 5, x0 = 3, y = 3
# jet color specturm

fig = plt.figure(figsize=(7, 7))

vis_2d_gaussian(X, Y, sigma_x = 5, sigma_y = 5, x0 = 3, y0 = 3, A = 255, cmap = 'jet')

plt.savefig('e5_output2.png')
