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
import shutil

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

def naming(data , agu_index):
	
	names_style = [
		
		"f_l_r",	#flip_left_right
		
		"f_r",		#flip_random
		
		"f_t_b",	#flip_top_bottom
		
		"g_d",		#gaussian_distortion
		
		"r_d",		#random_distortion
		
		"r_e",		#random_erasing
		
		"r_w_c",	#rotate_without_crop
		
		"sh",		#shear
		
		"sk",		#skew
		
		"sk_c",		#skew_corner
		
		"sk_l_r",	#skew_left_right
		
		"sk_t",		#skew_tilt
		
		"sk_t_b",	#skew_top_bottom
		
		"z",		#zoom
		
		"z_r",		#zoom_random
		
		"v",		#vignetting
		
		"s_a_p_n",	#salt_and_pepper_noise
		
		"c_s",		#color_shift
		
		] 

	enables = [

		data["flip_left_right"]["enable"],

		data["flip_random"]["enable"],

		data["flip_top_bottom"]["enable"],

		data["gaussian_distortion"]["enable"],

		data["random_distortion"]["enable"],

		data["random_erasing"]["enable"],

		data["rotate_without_crop"]["enable"],

		data["shear"]["enable"],

		data["skew"]["enable"],

		data["skew_corner"]["enable"],

		data["skew_left_right"]["enable"],

		data["skew_tilt"]["enable"],

		data["skew_top_bottom"]["enable"],

		data["zoom"]["enable"],

		data["zoom_random"]["enable"],

		data["vignetting"]["enable"],

		data["salt_and_pepper_noise"]["enable"],

		data["color_shift"]["enable"]

		]

	possible_name = []

	for item , enable in zip(names_style , enables):

		if enable :

			possible_name.append(item)

	names = []

	for index in agu_index :

		names.append(names_style[index])

	name = "__".join(names)

	return name

def agu_img(image , json_file_path , batch_size ) :

	if not os.path.isdir("./test"):

		os.mkdir("./test")

	cv2.imwrite("./test/image.png",image)	

	with open(json_file_path,'r') as json_file:
		
		data = json.load(json_file)
		json_file.close()

	first_time = True

	processed = []
	names =[]

	for x in range(batch_size):
	
		p = Augmentor.Pipeline("./test")
		
		agu1_1 = [

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

		enable1_1 = [

			data["flip_left_right"]["enable"],

			data["flip_random"]["enable"],

			data["flip_top_bottom"]["enable"],

			data["gaussian_distortion"]["enable"],

			data["random_distortion"]["enable"],

			data["random_erasing"]["enable"],

			data["rotate_without_crop"]["enable"],

			data["shear"]["enable"],

			data["skew"]["enable"],

			data["skew_corner"]["enable"],

			data["skew_left_right"]["enable"],

			data["skew_tilt"]["enable"],

			data["skew_top_bottom"]["enable"],

			data["zoom"]["enable"],

			data["zoom_random"]["enable"]

			]

		enable2_2 = [

			data["vignetting"]["enable"],

			data["salt_and_pepper_noise"]["enable"],

			data["color_shift"]["enable"]

			]

		length = enable1_1.count(True) + enable2_2.count(True)

		agu1 = []

		for item , enable in zip(agu1_1 , enable1_1) :

			if enable :

				agu1.append(item)


		agu_index1 = sorted(random.sample(range(0,length) , data["no_of_agu"]))

		names.append(naming(data , agu_index1))	#nm ch
		
		agu_index2 = []
	
		while agu_index1[len(agu_index1) - 1] > (len(agu1)-1) :
	
			agu_index2.append(agu_index1.pop() - len(agu1))
	
		for i in agu_index1 :
	
			agu1[i]
		
		batch_images = p.keras_generator(batch_size = 1, scaled = True, image_data_format = u'channels_last')
		batch_images1, labels = next(batch_images)
		batch_images1 = batch_images1[0] * 255 
	
		agu2_2 = [

			vignetting(batch_images1 , probability = data["vignetting"]["probability"] , px = data["vignetting"]["px"] , py = data["vignetting"]["py"]) ,
	
			salt_and_pepper_noise(batch_images1 , probability = data["salt_and_pepper_noise"]["probability"] , magnitude = data["salt_and_pepper_noise"]["magnitude"]) ,

			color_shift(batch_images1 , probability = data["color_shift"]["probability"] , color_shift_range = data["color_shift"]["color_shift_range"])
	
			]

		agu2 = []

		for item , enable in zip(agu2_2 , enable2_2) :

			if enable :

				agu2.append(item)

		for i in agu_index2 :
	
			batch_images1 = agu2[i]
		
		image = np.array(batch_images1, dtype= 'uint8')
	
		image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
		
		processed.append(image)

	shutil.rmtree("./test", ignore_errors=True)

	return processed ,names 	#nm ch

def agu_mul_img(images , json_file_path , total_batch_size) :

	com = total_batch_size // len(images)
	rem = total_batch_size % len(images)
	batch_sizes = [com]*len(images)
	batch_sizes[0 : (rem-1)] = [com + 1] * rem 
	batch_images = []
	batch_names = []

	for image , batch_size in zip(images , batch_sizes):

		processed , names = agu_img( image , json_file_path ,batch_size)

		batch_images.append(processed)
		batch_names.append(names)

	return batch_images , batch_names

json_file_path = "/home/guru/Desktop/work/projects/image_aug_-filtering/data.json"
'''batch_size = 200
img = "./download.jpeg"
image = cv2.imread(img)
processed , names = agu_img( image , json_file_path ,batch_size ) #nm ch

for x , image in enumerate(processed) :
	
	cv2.imwrite('./output/%s%s.png'%(x , names[x]),image)
'''
dir_path = "/home/guru/Desktop/work/projects/image_aug_-filtering/images"

path_images = [os.path.join(dir_path , f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path , f))]
total_batch_size = 200
images = []

for path in path_images :

	images.append(cv2.imread(path))
'''
for image in images:
	cv2.imshow('img',image)
	cv2.waitKey()
'''
batch_images , batch_names = agu_mul_img(images , json_file_path , total_batch_size)

for x , processed in enumerate(batch_images) :

	for y , image in enumerate(processed) :

		cv2.imwrite('./output/%s_%s_%s.png'%(x , y , batch_names[x][y]) , image)
