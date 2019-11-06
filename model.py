import csv
import cv2
import numpy as np
lines = []
with open('/home/workspace/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

car_images = []
steering_angles = []
car_images_side = []
steering_angles_side = []
steering_centers = 0
for line in lines:
    # create adjusted steering measurements for the side camera images
    correction = 0.21
    steering_center = float(line[3]) * 1.3
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # read in images from center, left and right cameras
#     img_center = np.expand_dims(cv2.cvtColor(cv2.imread(line[0]), cv2.COLOR_BGR2GRAY), axis=2)
#     img_left = np.expand_dims(cv2.cvtColor(cv2.imread(line[1]), cv2.COLOR_BGR2GRAY), axis=2)
#     img_right = np.expand_dims(cv2.cvtColor(cv2.imread(line[2]), cv2.COLOR_BGR2GRAY), axis=2)
    img_center = cv2.imread(line[0])
    img_left = cv2.imread(line[1])
    img_right = cv2.imread(line[2])
    
    car_images.append(img_center)
    steering_angles.append(steering_center)
    #1 水平翻转 0 垂直翻转 -1 水平垂直翻转
    flipped_image = cv2.flip(img_center,1)
    flipped_measurement = steering_center * -1.0
    car_images.append(flipped_image)
    steering_angles.append(flipped_measurement)
    car_images.append(img_left)
    steering_angles.append(steering_left)
    car_images.append(img_right)
    steering_angles.append(steering_right)
    
#Rotate image
augmented_car_images = []
augmented_steering_angles = []
for image,measurement in zip(car_images,steering_angles):
    augmented_car_images.append(image)
    augmented_steering_angles.append(measurement)
#     flipped_image = cv2.flip(image,1)#1 水平翻转 0 垂直翻转 -1 水平垂直翻转
#     flipped_measurement = measurement * -1.0
#     augmented_car_images.append(flipped_image)
#     augmented_steering_angles.append(flipped_measurement)


X_train = np.array(augmented_car_images)
y_train = np.array(augmented_steering_angles)


import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras.utils.visuallize_util import plot
from keras.utils import plot_model

model = Sequential()
model.add(Cropping2D(cropping=((75,25),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(24, 5, 5, border_mode='same',subsample = (2,2),activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, border_mode='same',subsample = (2,2),activation='relu'))
# model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, border_mode='same',subsample = (2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, border_mode='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
# model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.6))
model.add(Dense(1))

model.compile(optimizer = 'adam',loss = 'mse')
history_object = model.fit(X_train, y_train, validation_split = 0.1, batch_size=60,
                           nb_epoch=5, shuffle=True, verbose=1)
 
#plot_model(model,to_file = 'model_drop.png',show_shapes = True)
model.summary()
model.save('model.h5')

# from matplotlib.pyplot import plt
print(history_object.history.keys())
          
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
