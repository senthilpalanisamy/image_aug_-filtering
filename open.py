#importing required libraries
import Augmentor
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

def salt_andpepper_noise(image ,probability = 0.5 , magnitude = 0.004):

	if (probability < random.randint()):

		row,col,ch = image.shape
		s_vs_p = 0.5
		out = np.copy(image)
	    # Salt mode
		num_salt = np.ceil(magnitude * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
	    	    for i in image.shape]
		out[coords] = 1
	
	    # Pepper mode
		num_pepper = np.ceil(magnitude * image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
	            for i in image.shape]
		out[coords] = 0
	    
		return out

	else:

		return image

def vignetting(img ,probability = 0.5 , px = 0.1 , py = 0.1):

	if (probability > random.randint()):

		rows, cols = img.shape[:2]
		
		# generating vignette mask using Gaussian kernels
		kernel_x = cv2.getGaussianKernel(cols , cols * px)		
		kernel_y = cv2.getGaussianKernel(rows , rows * py)
		kernel = kernel_y * kernel_x.T
		mask = 255 * kernel / np.linalg.norm(kernel)
		output = np.copy(img)
		
		# applying the mask to each channel in the input image
		for i in range(3):
		    output[:,:,i] = output[:,:,i] * mask
		
		return output

	else:

		return img

gen = ImageDataGenerator(
	#featurewise_center=False,
    #samplewise_center=False,
    #featurewise_std_normalization=False,
    #samplewise_std_normalization=False,
    #zca_whitening=False,
    #zca_epsilon=1e-6,
    #rotation_range=0.,
    #width_shift_range=0.,
    #height_shift_range=0.,
    #shear_range=0.,
    #zoom_range=0.,
    channel_shift_range=0.,
    #fill_mode='nearest',
    #cval=0.,
    #horizontal_flip=False,
    #vertical_flip=False,
    #rescale=None,
    #preprocessing_function=None,
    data_format=K.image_data_format()
    )


path ='/home/guru/Desktop/Project_Prework/cap'
p = Augmentor.Pipeline(path)


p.black_and_white(probability = 0.5 , threshold = 120)

p.crop_by_size(probability = 0.5 , width = 1000 , height =1000 , centre = True)

p.crop_centre(probability = 0.5, percentage_area = 0.8 , randomise_percentage_area = False)

p.crop_random(probability =0.5 , percentage_area =0.8 , randomise_percentage_area = False )

p.flip_left_right(probability = 0.5)

p.flip_random(probability = 0.5) 

p.flip_top_bottom(probability = 0.5) 

p.gaussian_distortion(probability = 0.5 , grid_width = 15 , grid_height = 15 , magnitude = 100 ,
	corner= "bell" , method = "in" , mex = 0.5 , mey = 0.5 , sdx = 0.05 , sdy = 0.05) 

p.greyscale(probability = 0.5)

p.histogram_equalisation(probability = 0.5)

p.invert(probability = 0.5)

p.random_distortion(probability = 0.5 , grid_width = 15 , grid_height = 15 , magnitude = 100)

p.random_erasing (probability = 0.5 , rectangle_area =0.1 )

p.resize(probability =0.5 , width = 1000 , height = 1000, resample_filter=u'BICUBIC')

p.rotate (probability = 0.5 , max_left_rotation = 15 , max_right_rotation = 15)

p.rotate180 (probability = 0.5)

p.rotate270(probability = 0.5)

p.rotate90(probability = 0.5)

p.rotate_random_90(probability = 0.5)

p.rotate_without_crop(probability = 0.1 , max_left_rotation = 15 , max_right_rotation = 15, expand=False)

p.scale(probability = 0.5 , scale_factor = 1.2)

p.shear (probability = 0.5 , max_shear_left = 25 , max_shear_right = 25)

p.skew (probability = 0.5 , magnitude = 0.5) 

p.skew_corner (probability = 0.5 , magnitude=0.1)

p.skew_left_right (probability = 0.5 , magnitude=1)

p.skew_tilt (probability = 0.5 , magnitude=0.91)

p.skew_top_bottom (probability = 0.5 , magnitude=0.11)

p.zoom(probability = 0.5 , min_factor = 1.1 , max_factor = 2.0)

p.zoom_random (probability = 0.5 , percentage_area = 0.9 , randomise_percentage_area = False)

batch_images = p.keras_generator(batch_size = 1000, scaled = True, image_data_format = u'channels_last')
images = []

for img_process in batch_images:

	img_process1 = np.expand_dims(img_process , axis = 0)

	if (probability > random.randint()):

		img_process2 = gen.flow(img_process1)
		img_process3 = [next(img_process2)[0].astype(np.uint8)]
		images = np.append(images , img_process3 , axis = 0)

	else:

		images = np.append(images , img_process1 , axis = 0)
