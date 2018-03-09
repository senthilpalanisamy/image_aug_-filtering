#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import os
import numpy as np

#Types of agumentation applied

#Translation

def translate_image(img, trans_range = 10):
	
	tr_x = (2 * trans_range * np.random.uniform()) - trans_range
    tr_y = (2 * trans_range * np.random.uniform()) - trans_range
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    
    return img

#Rotation

def rotate_image(img, ang_range = 90):
	
	ang_rot = (2 * np.random.uniform(ang_range)) - ang_range
    rows, cols, ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    
    return img

#Brightness

def brighten_image(img, brightness_range = 1):
	
    random_bright = 0.25 + np.random.uniform(0, brightness_range)
    image1[:,:,2] = image1[:,:,2] * random_bright
        
    return image1

#Shear

def shear_image(img, shear_range = 10):

	pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5 + (2 * shear_range * np.random.uniform()) - shear_range
    pt2 = 20 + (2 * shear_range * np.random.uniform()) - shear_range
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img

#Prespective Transformation

def perspective_transform(X_img, offset = 15):
    
    
    img_size = (X_img.shape[1], X_img.shape[0])
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]), 						#left bottom
                          (0.15 * imshape[1], 0.20 * imshape[0]), 						#left top
                          (0.20 * imshape[1], 0.20 * imshape[0]), 						#right top
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype = np.int32)   #right bottom
    src = np.float32(vertices)
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0], 
                      [img_size[0] - offset, img_size[1]]])
    
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                    (X_img.shape[1], X_img.shape[0]),
                                    flags = cv2.INTER_LINEAR)
    
    return warped_img

#Gaussian Noise

def gaussian_noise(img):
	
	noisy = img.copy()
    m = (10000,10000,10000)  		#noise creation
    s = (10000,10000,10000)  		#noise visibility
    cv2.randn(noisy,m,s)
    
    return img + noisy

#Salt and Pepper Noise

def salt_andpepper_noise(image):

	row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004					#quantity of noise
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    
    return out

#Poissions Noise

def poissions_noise(image):

	vals = len(np.unique(image))
    print(vals)
    vals = 1.6 ** np.ceil(np.log2(vals)) 		#change the base value (1-2) to decrease the noise
    print(vals)
    noisy = np.random.poisson(image * vals) 
    
    return noisy

#Speckless Noise

def speckless_noise(image):

	row,col,ch = image.shape
    img = image.copy()
    gauss = np.random.randn(row,col,ch) * 0.5	# change the multiple to change the intensity of the noise
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + (20 * img * gauss)  		# change the multiple to change the intensity of the noise
    
    return noisy

#Vignetting

def vignetting(img):

	rows, cols = img.shape[:2]
	
	# generating vignette mask using Gaussian kernels
	kernel_x = cv2.getGaussianKernel(cols,100)		
	kernel_y = cv2.getGaussianKernel(rows,100)
	kernel = kernel_y * kernel_x.T
	mask = 255 * kernel / np.linalg.norm(kernel)
	output = np.copy(img)
	
	# applying the mask to each channel in the input image
	for i in range(3):
	    output[:,:,i] = output[:,:,i] * mask
	
	return output

#Resize

def resize_image(img, fx = 1.2, fy = 1.2, interpolation = cv2.INTER_LINEAR):

	#types of interpolation INTER_LINEAR, INTER_AREA, INTER_CUBIC
	img_scaled = cv2.resize(img, None, fx = fx, fy = fy, interpolation = interpolation)

	return img_scaled

#Histogram Equalization

def histogram_equalization(img):

	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	
	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	return img_output

#Blurring

def blur_image(img, blur_type = 'gaussain'):

	#blur type selection
	if (blur_type == 'averaging'):

		blur = cv2.blur(img,(5,5))
	
	elif(blur_type == 'gaussain'):

		blur = cv2.GaussianBlur(img,(5,5),0)

	elif(blur_type == 'median'):

		blur = cv2.medianBlur(img,5)

	elif(blur_type == 'bilateral'):

		blur = cv2.bilateralFilter(img,9,75,75)

	return blur

#Channel Shift

def channel_shift(img, shift_type = 'HSV'):

	#type of channel shift
	return eval('cv2.cvtColor(image, cv2.COLOR_BGR2' + shift_type + ')')

#Mirroring

def mirror_image(img):

	# copy image to display all 4 variations
	horizontal_img = img.copy()
	vertical_img = img.copy()
	both_img = img.copy()
	 
	# flip img horizontally, vertically
	horizontal_img = cv2.flip( img, 0 )
	vertical_img = cv2.flip( img, 1 )

	# and both axes with flip()
	both_img = cv2.flip( img, -1 )
	
	return [horizontal_img , vertical_img , both_img]