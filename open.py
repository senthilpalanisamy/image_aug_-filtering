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

def salt_and_pepper_noise(images ,probability = 0.5 , magnitude = 0.004):

	sp_img = []
	for image in images :
		if (probability < random.random()):
	
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
		    
			sp_img.append(out)
	
		else:
	
			sp_img.append(image)

	return sp_img

def vignetting(images , probability = 0.5 , px = 0.25 , py = 0.25):

	v_img = []
	for img in images:

		if (probability < random.random()):


			#print(random.random())
			rows, cols = img.shape[:2]
			
			# generating vignette mask using Gaussian kernels
			kernel_x = cv2.getGaussianKernel(cols , 200)		
			kernel_y = cv2.getGaussianKernel(rows , rows * py)
			kernel = kernel_y * kernel_x.T
			mask = ((rows+cols)//6) * kernel / np.linalg.norm(kernel)
			output = np.copy(img)
			
			# applying the mask to each channel in the input image
			for i in range(3):
			    output[:,:,i] = output[:,:,i] * mask
			
			v_img.append(output)
	
		else:
	
			v_img.append(img)

	return v_img


def color_shift(batch_images , probability = 0.5 , color_shift_range = 15):

	gen = ImageDataGenerator(
	    channel_shift_range = color_shift_range,
	    data_format=K.image_data_format()
	    )

	images = []


	for img_process in batch_images:

		
		img_process1 = np.expand_dims(img_process , axis = 0)

		if (probability > random.random()):
	
			img_process2 = gen.flow(img_process1)
			img_process3 = [next(img_process2)[0].astype(np.uint8)]
			
			images.append(img_process3)
		else:
			images.append(img_process1)
	return images

path ='/home/guru/Desktop/Project_Prework/cap'
p = Augmentor.Pipeline(path)

#p.black_and_white(probability = 0.5 , threshold = 120)
#
#p.crop_by_size(probability = 0.5 , width = 1000 , height =1000 , centre = True)
#
#p.crop_centre(probability = 0.5, percentage_area = 0.8 , randomise_percentage_area = False)
#
#p.crop_random(probability =0.5 , percentage_area =0.9 , randomise_percentage_area = False )

p.flip_left_right(probability = 0.5)

p.flip_random(probability = 0.5) 

p.flip_top_bottom(probability = 0.5) 

p.gaussian_distortion(probability = 0.5 , grid_width = 5 , grid_height = 5 , magnitude = 10 ,
	corner= "bell" , method = "in" , mex = 0.5 , mey = 0.5 , sdx = 0.05 , sdy = 0.05) 

#p.greyscale(probability = 0.5)

#p.histogram_equalisation(probability = 0.5)
#
#p.invert(probability = 0.5)

p.random_distortion(probability = 0.5 , grid_width = 5 , grid_height = 5 , magnitude = 10)
'''
(probability = 0.5 , grid_width = 5 , grid_height = 5 , magnitude = 10)
'''
p.random_erasing (probability = 0.5 , rectangle_area =0.15 )

#p.resize(probability =0.5 , width = 1000 , height = 1000, resample_filter=u'BICUBIC')
#
#p.rotate (probability = 0.5 , max_left_rotation = 15 , max_right_rotation = 15)
#
#p.rotate180 (probability = 0.5)
#
#p.rotate270(probability = 0.5)
#
#p.rotate90(probability = 0.5)
#
#p.rotate_random_90(probability = 0.5)

p.rotate_without_crop(probability = 0.1 , max_left_rotation = 15 , max_right_rotation = 15, expand=False)

#p.scale(probability = 0.5 , scale_factor = 1.2)

p.shear (probability = 0.5 , max_shear_left = 25 , max_shear_right = 25)

p.skew (probability = 0.5 , magnitude = 0.5) 

p.skew_corner (probability = 0.5 , magnitude=0.1)

p.skew_left_right (probability = 0.5 , magnitude=.5)

p.skew_tilt (probability = 0.5 , magnitude=0.91)

p.skew_top_bottom (probability = 0.5 , magnitude=0.11)

p.zoom(probability = 0.5 , min_factor = 1.1 , max_factor = 1.2)

p.zoom_random (probability = 0.5 , percentage_area = 0.9 , randomise_percentage_area = False)

batch_images = p.keras_generator(batch_size = 100, scaled = True, image_data_format = u'channels_last')
#print(batch_images)
batch_images1, labels = next(batch_images)
#print(batch_images1)
#batch_images1 = cv2.cvtColor(batch_images1 , cv2.COLOR_RGB2BGR)
batch_images1 = batch_images1 * 255

batch_images2 = vignetting(batch_images1 , probability = 0.5 , px = 0.25 , py = 0.25)

batch_images3 = salt_and_pepper_noise(batch_images2 ,probability = 0.5 , magnitude = 0.004)

batch_images4 = color_shift(batch_images = batch_images3 , probability = 0.5 , color_shift_range = 15)

for i, image in enumerate(batch_images4):
	image = np.array(image, dtype= 'uint8')
	print(image.shape)
	image = cv2.cvtColor(image[0] , cv2.COLOR_RGB2BGR)
	#print(image.shape)
	cv2.imwrite('./output/img%s.jpg'%i,image)
