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
import itertools

#Salt and Pepper Noise

def salt_and_pepper_noise(image , probability = 0.5 , magnitude = 0.004):

	#generating a random number for checking the probability

	if (probability > random.random()):

		row,col,ch = image.shape
		
		#the ratio of salt noise

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

#Vignetting

def vignetting(img , probability = 0.5 , px = 0.25 , py = 0.25):
	
	#generating a random number for checking the probability

	if (probability > random.random()):

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

#Color Shift

def color_shift(img_process , probability = 0.5 , color_shift_range = 50):

	#selecting the required operations in keras lib

	gen = ImageDataGenerator(
	    channel_shift_range = color_shift_range,
	    data_format=K.image_data_format()
	    )
	
	#expanding the dimension so that the keras lib can process a single image
	
	img_process1 = np.expand_dims(img_process , axis = 0)

	#generating a random number for checking the probability

	if (probability > random.random()):

		
		img_process2 = gen.flow(img_process1)
		
		img_process3 = [next(img_process2)[0].astype(np.uint8)]
		
		images = img_process3[0]
	
	else:
	
		images = img_process1[0]
	
	images = np.array(images, dtype= 'uint8')

	return images

def possible_functions(data) :

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

	probabilities = [

		data["flip_left_right"]["probability"],

		data["flip_random"]["probability"],

		data["flip_top_bottom"]["probability"],

		data["gaussian_distortion"]["probability"],

		data["random_distortion"]["probability"],

		data["random_erasing"]["probability"],

		data["rotate_without_crop"]["probability"],

		data["shear"]["probability"],

		data["skew"]["probability"],

		data["skew_corner"]["probability"],

		data["skew_left_right"]["probability"],

		data["skew_tilt"]["probability"],

		data["skew_top_bottom"]["probability"],

		data["zoom"]["probability"],

		data["zoom_random"]["probability"],

		data["vignetting"]["probability"],

		data["salt_and_pepper_noise"]["probability"],

		data["color_shift"]["probability"]

		]

	functions = [
		
		"flip_left_right",
		
		"flip_random",
		
		"flip_top_bottom",
		
		"gaussian_distortion",
		
		"random_distortion",
		
		"random_erasing",
		
		"rotate_without_crop",
		
		"shear",
		
		"skew",	
		
		"skew_corner",
		
		"skew_left_right",
		
		"skew_tilt",
		
		"skew_top_bottom",
		
		"zoom",	
		
		"zoom_random",
		
		"vignetting",
		
		"salt_and_pepper_noise",
		
		"color_shift",
		
		]

	possible = []

	probability = []

	for item , pro , enable in zip(functions , probabilities , enables):

		if enable :

			possible.append(item)

			probability.append(pro)

	pro_sum = sum(probability)

	cal_probability = map(lambda x : x / pro_sum , probability)

	return possible , cal_probability

#Functions Selection 

def function_selection(data , image_no):

	p = Augmentor.Pipeline("./test")

	possible , probability = possible_functions(data)

	ran = random.randint(1 , data["no_of_agu"])

	possible = list(possible)

	probability = list(probability)

	sel_fun = np.random.choice(possible , ran , replace = False , p = probability)

	names = [image_no]

	if "flip_left_right" in sel_fun :

		p.flip_left_right(probability = 1)

		names.append("f_l_r")

	if "flip_random" in sel_fun :

		p.flip_random(probability = 1)

		names.append("f_r")

	if "flip_top_bottom" in sel_fun :

		p.flip_top_bottom(probability = 1)

		names.append("f_t_b")

	if "gaussian_distortion" in sel_fun :

		p.gaussian_distortion(probability = 1 , grid_width = data["gaussian_distortion"]["grid_width"] , grid_height = data["gaussian_distortion"]["grid_height"] , magnitude = data["gaussian_distortion"]["magnitude"] , corner= "bell" , method = "in" , mex = 0.5 , mey = 0.5 , sdx = 0.05 , sdy = 0.05)

		names.append("g_d")

	if "random_distortion" in sel_fun :

		p.random_distortion(probability = 1  , grid_width = data["random_distortion"]["grid_width"] , grid_height = data["random_distortion"]["grid_height"] , magnitude = data["random_distortion"]["magnitude"]) 

		names.append("r_d")

	if "random_erasing" in sel_fun :

		p.random_erasing (probability = 1 , rectangle_area = data["random_erasing"]["rectangle_area"] )

		names.append("r_e")

	if "rotate_without_crop" in sel_fun :

		p.rotate_without_crop(probability = 1 , max_left_rotation = data["rotate_without_crop"]["max_left_rotation"] , max_right_rotation = data["rotate_without_crop"]["max_right_rotation"], expand=False) 

		names.append("r_w_c")

	if "shear" in sel_fun :

		p.shear (probability = 1 , max_shear_left = data["shear"]["max_shear_left"] , max_shear_right = data["shear"]["max_shear_right"])

		names.append("sh")

	if "skew" in sel_fun :

		p.skew (probability = 1 , magnitude = data["skew"]["magnitude"]) 

		names.append("sk")

	if "skew_corner" in sel_fun :

		p.skew_corner (probability = 1 , magnitude = data["skew_corner"]["magnitude"])

		names.append("sk_c")

	if "skew_left_right" in sel_fun :

		p.skew_left_right (probability = 1 , magnitude = data["skew_left_right"]["magnitude"])

		names.append("sk_l_r")

	if "skew_tilt" in sel_fun :

		p.skew_tilt (probability = 1 , magnitude = data["skew_tilt"]["magnitude"])

		names.append("sk_t")

	if "skew_top_bottom" in sel_fun :

		p.skew_top_bottom (probability = 1 , magnitude = data["skew_top_bottom"]["magnitude"] )

		names.append("sk_t_b")

	if "zoom" in sel_fun :

		p.zoom(probability = 1 , min_factor = data["zoom"]["min_factor"] , max_factor = data["zoom"]["max_factor"])

		names.append("z")

	if "zoom_random" in sel_fun :

		p.zoom_random (probability = 1 , percentage_area = data["zoom_random"]["percentage_area"] , randomise_percentage_area = False)

		names.append("z_r")

	batch_images = p.keras_generator(batch_size = 1, scaled = True, image_data_format = u'channels_last')
	
	batch_images1, labels = next(batch_images)
	
	batch_images1 = batch_images1[0] * 255 

	if "vignetting" in sel_fun :

		batch_images1 = vignetting(batch_images1 , probability = 1 , px = data["vignetting"]["px"] , py = data["vignetting"]["py"])

		names.append("v")
		
	if "salt_and_pepper_noise" in sel_fun :

		batch_images1 = salt_and_pepper_noise(batch_images1 , probability = 1 , magnitude = data["salt_and_pepper_noise"]["magnitude"])

		names.append("s_a_p_n")

	if "color_shift" in sel_fun :

		batch_images1 = color_shift(batch_images1 , probability = 1 , color_shift_range = data["color_shift"]["color_shift_range"])

		names.append("c_s")

	image = np.array(batch_images1, dtype= 'uint8')

	image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)

	name = "__".join(names)

	return image , name

def agu_img(image , json_file_path , batch_size = 100 ,image_no = "1") :

	if not os.path.isdir("./test"):

		os.mkdir("./test")

	cv2.imwrite("./test/image.png" , image)	

	with open(json_file_path,'r') as json_file:
		
		data = json.load(json_file)
		
		json_file.close()

	processed = []
	
	names =[]

	for x in range(batch_size):

		image , name = function_selection(data , image_no)

		names.append(name)
		
		processed.append(image)

	shutil.rmtree("./test", ignore_errors=True)

	return processed ,names 	#nm ch

def agu_mul_img(images , json_file_path , total_batch_size) :

	com = total_batch_size // len(images)
	
	rem = total_batch_size % len(images)
	
	batch_sizes = [com]*len(images)
	
	if rem != 0 :
	
		batch_sizes[0 : (rem-1)] = [com + 1] * rem 
	
	batch_images = []
	
	batch_names = [] 

	image_no = 1

	for image , batch_size in zip(images , batch_sizes):

		processed , names = agu_img( image , json_file_path ,batch_size , image_no = str(image_no))

		image_no = image_no + 1

		batch_images.append(processed)
	
		batch_names.append(names)

	processed_img = [img for batch in batch_images for img in batch]

	processed_nam = [nam for batch in batch_names for nam in batch]

	return processed_img , processed_nam

json_file_path = "./data.json"

'''
batch_size = 200

img = "./download.jpeg"

image = cv2.imread(img)

processed , names = agu_img( image , json_file_path ,batch_size ) #nm ch

for x , image in enumerate(processed) :
	
	cv2.imwrite('./output/%s%s.png'%(x , names[x]),image)
'''

dir_path = "./images"

path_images = [os.path.join(dir_path , f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path , f))]

total_batch_size = 200

images = []

for path in path_images :

	images.append(cv2.imread(path))

batch_images , batch_names = agu_mul_img(images , json_file_path , total_batch_size)

for x , image  in enumerate(batch_images) :

	cv2.imwrite('./output/img%s__%s.png'%(batch_names[x] , x) , image)
