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
import json
#import shutil

def salt_and_pepper_noise(image ,probability = 0.5 , magnitude = 0.004):

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
	    
		sp_img = out

	else:

		sp_img = image

	return sp_img

def vignetting(img , probability = 0.5 , px = 0.25 , py = 0.25):

	if (probability < random.random()):

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
		
		v_img = output

	else:

		v_img = img
	
	return v_img


def color_shift(img_process , probability = 0.5 , color_shift_range = 50):

	gen = ImageDataGenerator(
	    channel_shift_range = color_shift_range,
	    data_format=K.image_data_format()
	    )
		
	img_process1 = np.expand_dims(img_process , axis = 0)

	if (probability > random.random()):

		img_process2 = gen.flow(img_process1)
		img_process3 = [next(img_process2)[0].astype(np.uint8)]
		
		images = img_process3[0]
	
	else:
	
		images = img_process1[0]
	
	images = np.array(images, dtype= 'uint8')

	return images

def agu_img(image , json_file_path ,batch_size ) :

	if not os.path.isdir("./test"):
		os.mkdir("./test")
	img = cv2.imread(image)
	cv2.imwrite("./test/image.png",img)	

	with open(json_file_path,'r') as json_file:
		
		data = json.load(json_file)
		json_file.close()
	processed = []

	for x in range(batch_size):
	
		p = Augmentor.Pipeline("./test")
	
		agu1 = [
			p.flip_left_right(probability = data["flip_left_right"]["probability"]),
			
			p.flip_random(probability = data["flip_random"]["probability"]),
			
			p.flip_top_bottom(probability = data["flip_top_bottom"]["probability"]), 
			
			p.gaussian_distortion(probability = data["gaussian_distortion"]["probability"] , grid_width = data["gaussian_distortion"]["grid_width"] , grid_height = data["gaussian_distortion"]["grid_height"] , magnitude = data["gaussian_distortion"]["magnitude"] , corner= "bell" , method = "in" , mex = 0.5 , mey = 0.5 , sdx = 0.05 , sdy = 0.05) ,
			
			p.random_distortion(probability = data["random_distortion"]["probability"]  , grid_width = data["random_distortion"]["grid_width"] , grid_height = data["random_distortion"]["grid_height"] , magnitude = data["random_distortion"]["magnitude"]) ,
			
			p.random_erasing (probability = data["random_erasing"]["probability"] , rectangle_area = data["random_erasing"]["rectangle_area"] ) ,
			
			p.rotate_without_crop(probability = data["rotate_without_crop"]["probability"] , max_left_rotation = data["rotate_without_crop"]["max_left_rotation"] , max_right_rotation = data["rotate_without_crop"]["max_right_rotation"], expand=False) ,
			
			p.shear (probability = data["shear"]["probability"] , max_shear_left = data["shear"]["max_shear_left"] , max_shear_right = data["shear"]["max_shear_right"]) ,
			
			p.skew (probability = data["skew"]["probability"] , magnitude = data["skew"]["magnitude"]) ,
			
			p.skew_corner (probability = data["skew_corner"]["probability"] , magnitude = data["skew_corner"]["magnitude"]) ,
			
			p.skew_left_right (probability = data["skew_left_right"]["probability"] , magnitude = data["skew_left_right"]["magnitude"]) ,
			
			p.skew_tilt (probability = data["skew_tilt"]["probability"] , magnitude = data["skew_tilt"]["magnitude"]) ,
			
			p.skew_top_bottom (probability = data["skew_top_bottom"]["probability"] , magnitude = data["skew_top_bottom"]["magnitude"] ) ,
			
			p.zoom(probability = data["zoom"]["probability"] , min_factor = data["zoom"]["min_factor"] , max_factor = data["zoom"]["max_factor"]) ,
			
			p.zoom_random (probability = data["zoom_random"]["probability"] , percentage_area = data["zoom_random"]["percentage_area"] , randomise_percentage_area = False)
			
			]
	
		agu_index1 = []
	
		for i in range(data["no_of_agu"]):
	
			agu_index1.append(random.randint(0,17))
	
		agu_index1 = sorted(list(set(agu_index1))) 
	
		agu_index2 = []
	
		while agu_index1[len(agu_index1) - 1] > 14 :
	
			agu_index2.append(agu_index1.pop() - 15)
	
		for i in agu_index1 :
	
			agu1[i]
		
		batch_images = p.keras_generator(batch_size = 1, scaled = True, image_data_format = u'channels_last')
		batch_images1, labels = next(batch_images)
		batch_images1 = batch_images1[0] * 255 #che
	
		agu2 = [
			vignetting(batch_images1 , probability = data["vignetting"]["probability"] , px = data["vignetting"]["px"] , py = data["vignetting"]["py"]) ,
			
			salt_and_pepper_noise(batch_images1 , probability = data["salt_and_pepper_noise"]["probability"] , magnitude = data["salt_and_pepper_noise"]["magnitude"]) ,
	
			color_shift(batch_images1 , probability = data["color_shift"]["probability"] , color_shift_range = data["color_shift"]["color_shift_range"])
			
			]
		
		for i in agu_index2 :
	
			batch_images1 = agu2[i]
		
		image = np.array(batch_images1, dtype= 'uint8')
	
		image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
		
		processed.append(image)
	
	shutil.rmtree(dest, ignore_errors=True)
	return processed
		
json_file_path = "/home/guru/Desktop/work/projects/image_aug_-filtering/data.json"
image = "/home/guru/Desktop/work/projects/image_aug_-filtering/download.jpeg"
batch_size = 200

processed = agu_img( image , json_file_path ,batch_size )

for i , image in enumerate(processed) :

	cv2.imwrite('./output/img%s.png'%i,image)