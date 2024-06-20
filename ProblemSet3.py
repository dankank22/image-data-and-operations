{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "36ab91de-8903-427d-94ed-c3e926925206",
   "metadata": {},
   "source": [
    "# Lab 3 Report\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import numpy as np\n",
    "\n",
    "from PIL import Image \n",
    "%matplotlib inline\n",
    "\n",
    "from IPython.display import Image as ipyimage #For displaying images in colab jupyter cell\n",
    "\n",
    "### Exercise 1: Generalized function for subsetting pixels\n",
    "\n",
    "ipyimage('lab3_exercise1.PNG', width = 1000)\n",
    "\n",
    "e1_img = Image.open('sample_image_1.jpg').convert('L')\n",
    "e1_img = np.array(e1_img)     # 8-bit code grayscale\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e1_img, cmap = 'gray')\n",
    "\n",
    "def subset_pixels(image, min_pixel_depth, max_pixel_depth, replacement_val):\n",
    "     \n",
    "  img_gray_copy = image.copy()            # Copy the image to a new variable to avoid operating on a same image\n",
    "  \n",
    "  img_gray_copy[img_gray_copy < min_pixel_depth] = replacement_val    #Apply boolean mask\n",
    "  img_gray_copy[img_gray_copy > max_pixel_depth] = replacement_val\n",
    "    \n",
    "  return img_gray_copy                    # Return the new image\n",
    "\n",
    "# min_pixel_depth = 50, max_pixel_depth = 250, replacement_val = 100\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "e1_output1 = subset_pixels(image = e1_img, min_pixel_depth = 50, max_pixel_depth = 250, replacement_val = 100)\n",
    "\n",
    "plt.imshow(e1_output1, cmap = 'gray', vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e1_output1.png')\n",
    "\n",
    "# min_pixel_depth = 100, max_pixel_depth = 200, replacement_val = 30\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "e1_output2 = subset_pixels(image = e1_img, min_pixel_depth = 100, max_pixel_depth = 200, replacement_val = 30)\n",
    "\n",
    "plt.imshow(e1_output2, cmap = 'gray', vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e1_output2.png')\n",
    "\n",
    "# min_pixel_depth = 50, max_pixel_depth = 255, replacement_val = 255\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "e1_output3 = subset_pixels(image = e1_img, min_pixel_depth = 50, max_pixel_depth = 255, replacement_val = 255)\n",
    "\n",
    "plt.imshow(e1_output3, cmap = 'gray', vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e1_output3.png')\n",
    "\n",
    "### Exercise 2: Thumbnail generator function\n",
    "\n",
    "ipyimage('lab3_exercise2.PNG', width = 1000)\n",
    "\n",
    "e2_img = mpimg.imread('sample_image_2.jpg')\n",
    "e2_img = np.array(e2_img)     \n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e2_img)\n",
    "\n",
    "\n",
    "    \n",
    "def create_thumbnail(image, downsampling_rate):\n",
    "    \n",
    "    height, width, channel = image.shape\n",
    "    height_copy = height // downsampling_rate\n",
    "    width_copy = width // downsampling_rate\n",
    "\n",
    "    output_img = np.zeros((height_copy, width_copy, 3), dtype=np.uint8) # converts to 8 bit (0-255)\n",
    "\n",
    "    # i and j iterates through the image based on the downsampling rate\n",
    "    for i in range(height_copy):\n",
    "        for j in range(width_copy):\n",
    "            i_0 = i * downsampling_rate\n",
    "            j_0 = j * downsampling_rate\n",
    "            i_x = i_0 + downsampling_rate\n",
    "            j_x = j_0 + downsampling_rate\n",
    "    # segment finds the 2d array within the given start and end i and j values\n",
    "            segment = image[i_0:i_x, j_0:j_x, :]\n",
    "            avg_color = np.mean(segment, axis=(0, 1))\n",
    "            output_img[i, j, :] = avg_color.astype(np.uint8)\n",
    "    \n",
    "    return output_img\n",
    "\n",
    "\n",
    "\n",
    "# downsampling rate x5\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "e2_output1 = create_thumbnail(image = e2_img, downsampling_rate = 5)\n",
    "\n",
    "plt.imshow(e2_output1, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e2_output1.png')\n",
    "\n",
    "# downsampling rate x10\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "e2_output2 = create_thumbnail(image = e2_img, downsampling_rate = 10)\n",
    "\n",
    "plt.imshow(e2_output2, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e2_output2.png')\n",
    "\n",
    "# downsampling rate x20\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "e2_output3 = create_thumbnail(image = e2_img, downsampling_rate = 20)\n",
    "\n",
    "plt.imshow(e2_output3, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e2_output3.png')\n",
    "\n",
    "## Extra credit: Code efficiency\n",
    "### Achieve a runtime speed of < 100ms\n",
    "\n",
    "timeit -n 1 -r 7 e2_output2 = create_thumbnail(image = e2_img, downsampling_rate = 5)\n",
    "\n",
    "### Exercise 3: Generalized image blender function\n",
    "\n",
    "ipyimage('lab3_exercise3.PNG', width = 1000)\n",
    "\n",
    "e3_img1 = mpimg.imread('sample_image_1.jpg')\n",
    "e3_img2 = mpimg.imread('sample_image_2.jpg')\n",
    "e3_img3 = mpimg.imread('sample_image_3.jpg')\n",
    "e3_img4 = mpimg.imread('sample_image_4.jpg')\n",
    "e3_img5 = mpimg.imread('sample_image_5.jpg')\n",
    "\n",
    "e3_img1 = np.array(e3_img1)     \n",
    "e3_img2 = np.array(e3_img2)     \n",
    "e3_img3 = np.array(e3_img3)     \n",
    "e3_img4 = np.array(e3_img4)     \n",
    "e3_img5 = np.array(e3_img5)     \n",
    "\n",
    "fig = plt.figure(figsize=(15, 15))\n",
    "\n",
    "plt.subplot(1,5,1)\n",
    "plt.imshow(e3_img1, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.subplot(1,5,2)\n",
    "plt.imshow(e3_img2, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.subplot(1,5,3)\n",
    "plt.imshow(e3_img3, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.subplot(1,5,4)\n",
    "plt.imshow(e3_img4, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.subplot(1,5,5)\n",
    "plt.imshow(e3_img5, vmin = 0, vmax = 255)\n",
    "\n",
    "def blend_images(image_list, weight_list):\n",
    "\n",
    "    #initialise variables to store blended values for each colour channel\n",
    "    \n",
    "    blended_red = 0\n",
    "    blended_blue = 0\n",
    "    blended_green = 0\n",
    "    \n",
    "    # iterate through each image and multiply the pixels by the respective weight and added to the blended result\n",
    "   \n",
    "    for index, image in enumerate(image_list):\n",
    "        blended_red += image[:,:,0]*weight_list[index] \n",
    "        blended_blue += image[:,:,2]*weight_list[index] \n",
    "        blended_green += image[:,:,1]*weight_list[index] \n",
    "    \n",
    "    blended_img = np.stack([blended_red, blended_green, blended_blue], axis = 2)\n",
    "    blended_img = blended_img.astype(int) # multipolication with the float weights can create float values in our \n",
    "    \n",
    "    return blended_img\n",
    "\n",
    "# Blend all 5 images with equal weights\n",
    "\n",
    "e3_part1_image_list =[e3_img1, e3_img2, e3_img3, e3_img4, e3_img5]\n",
    "e3_part1_weight_list = [0.2, 0.2, 0.2, 0.2, 0.2]\n",
    "\n",
    "e3_output1 = blend_images(image_list = e3_part1_image_list, weight_list = e3_part1_weight_list)\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e3_output1, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e3_output1.png')\n",
    "\n",
    "# Blend first 3 images with different weights\n",
    "\n",
    "e3_part2_image_list =[e3_img1, e3_img2, e3_img3]\n",
    "e3_part2_weight_list = [0.2, 0.3, 0.5]\n",
    "\n",
    "e3_output2 = blend_images(image_list = e3_part2_image_list, weight_list = e3_part2_weight_list)\n",
    "\n",
    "fig = plt.figure(figsize=(5, 5))\n",
    "\n",
    "plt.imshow(e3_output2, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e3_output2.png')\n",
    "\n",
    "### Exercise 4: Image rotation function\n",
    "\n",
    "ipyimage('lab3_exercise4.PNG', width = 1000)\n",
    "\n",
    "e4_img = mpimg.imread('sample_image_2.jpg')\n",
    "e4_img = np.array(e4_img)     \n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e4_img, vmin = 0, vmax = 255)\n",
    "\n",
    "def rotate_image(image, rotate_angle):\n",
    " \n",
    "    img_copy = image.copy()\n",
    "    \n",
    "    if rotate_angle == 90:   #rotate by 90 degrees\n",
    "        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=1) # axis 1 flips horizontally\n",
    "        return img_copy\n",
    "\n",
    "    elif rotate_angle == 270:    # rotate by 270 degrees\n",
    "        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=0) # axis 0 flips vertically\n",
    "        return img_copy\n",
    "\n",
    "    elif rotate_angle == 180:    #rotate by 180 degrees\n",
    "        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=1)\n",
    "        img_copy = np.flip(img_copy.transpose(1, 0, 2), axis=1)\n",
    "        return img_copy\n",
    "\n",
    "    else:                        #rotate by 0 degrees\n",
    "        return img_copy\n",
    "\n",
    "# Rotate the image by 0 degrees - This should result in identity\n",
    "\n",
    "e4_output1 = rotate_image(image = e4_img, rotate_angle = 0)\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e4_output1, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e4_output1.png')\n",
    "\n",
    "# Rotate the image by 90 degrees\n",
    "\n",
    "e4_output2 = rotate_image(image = e4_img, rotate_angle = 90)\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e4_output2, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e4_output2.png')\n",
    "\n",
    "# Rotate the image by 180 degrees\n",
    "\n",
    "e4_output3 = rotate_image(image = e4_img, rotate_angle = 180)\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e4_output3, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e4_output3.png')\n",
    "\n",
    "# Rotate the image by 270 degrees\n",
    "\n",
    "e4_output4 = rotate_image(image = e4_img, rotate_angle = 270)\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "plt.imshow(e4_output4, vmin = 0, vmax = 255)\n",
    "\n",
    "plt.savefig('e4_output4.png')\n",
    "\n",
    "## Extra credit: Code efficiency\n",
    "### Achieve a runtime speed of < 5ms\n",
    "\n",
    "timeit -n 1 -r 7 e4_output2 = rotate_image(image = e4_img, rotate_angle = 90)\n",
    "\n",
    "### Exercise 5: 2D Gaussian image generator\n",
    "\n",
    "ipyimage('lab3_exercise5.PNG', width = 1000)\n",
    "\n",
    "x_range = np.arange(-25, 25, 1)\n",
    "y_range = np.arange(-25, 25, 1)[::-1]\n",
    "\n",
    "X, Y = np.meshgrid(x_range, y_range) # More detail on documentation \n",
    "                                     # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html\n",
    "\n",
    "print(X, X.shape) # Set of x-coordinates in the function domain     \n",
    "\n",
    "print(Y, Y.shape) # Set of y-coordinates in the function domain   \n",
    "\n",
    "def vis_2d_gaussian(X, Y, sigma_x, sigma_y, x0, y0, A, cmap):\n",
    "\n",
    "    Z = A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2)))\n",
    "    \n",
    "    # Create figure and axis for the plot\n",
    "    figure, axis = plt.subplots(figsize=(7, 7))\n",
    "    \n",
    "    # Display the 2D Gaussian\n",
    "    cax = axis.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap)\n",
    "    figure.colorbar(cax)\n",
    "   \n",
    "\n",
    "\n",
    "\n",
    "# First parameter set: sigma_x = 10, sigma_y = 10, x0 = 0, y0 = 0\n",
    "# cmap = 'gray'\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "vis_2d_gaussian(X, Y, sigma_x = 10, sigma_y = 10, x0 = 0, y0 = 0, A = 255, cmap = 'gray')\n",
    "\n",
    "plt.savefig('e5_output1.png')\n",
    "\n",
    "# First parameter set\n",
    "# cmap = 'jet'\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "vis_2d_gaussian(X, Y, sigma_x = 10, sigma_y = 10, x0 = 0, y0 = 0, A = 255, cmap = 'jet')\n",
    "\n",
    "plt.savefig('e5_output2.png')\n",
    "\n",
    "# Second parameter set: sigma_x = 5, sigma_y = 5, x0 = 3, y = 3\n",
    "# grayscale color specturm\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "vis_2d_gaussian(X, Y, sigma_x = 5, sigma_y = 5, x0 = 3, y0 = 3, A = 255, cmap = 'gray')\n",
    "\n",
    "plt.savefig('e5_output1.png')\n",
    "\n",
    "# Second parameter set: sigma_x = 5, sigma_y = 5, x0 = 3, y = 3\n",
    "# jet color specturm\n",
    "\n",
    "fig = plt.figure(figsize=(7, 7))\n",
    "\n",
    "vis_2d_gaussian(X, Y, sigma_x = 5, sigma_y = 5, x0 = 3, y0 = 3, A = 255, cmap = 'jet')\n",
    "\n",
    "plt.savefig('e5_output2.png')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
