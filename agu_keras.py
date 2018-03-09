from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2
from scipy import ndimage

img_path = "/home/guru/Desktop/Project_Prework/cap/cap.JPG"
img1 = image.load_img(img_path)
#img = img.astype('float32')
gen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
s = np.shape(img1)
img = np.expand_dims(image.load_img(img_path),axis = 0)

a = gen.flow(img)
print(np.shape(img),s)
b = [next(a)[0].astype(np.uint8) for i in range(5)]

for i in range(5):
	plt.subplot(5,1,i+1) , plt.imshow(b[i])

plt.imshow(img1)

plt.show()