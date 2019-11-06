import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg

angle_correct = 0.23
      
samples = []
with open('/home/workspace/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

images = []
angles = []
for sample in samples:
    center_angle = float(sample[3])
    for i in range(3):
        name = '/home/workspace//IMG/'+sample[i].split('/')[-1]
        img = cv2.imread(name)
        #img = cv2.resize(img, (80,40))
        #img = img[:,:,2] #choose S channel          
        images.append(img)
        images.append(cv2.flip(img,1))                                      
                    
        if(i== 0):                        
            angles.append(center_angle)
            angles.append(center_angle* -1)
        if(i == 1):
            angles.append(center_angle + angle_correct)
            angles.append((center_angle + angle_correct)* -1)                     
        if(i== 2):
            angles.append(center_angle - angle_correct)
            angles.append((center_angle - angle_correct)* -1)
                        
            # trim image to only see section with road
X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Lambda, Convolution2D, Dropout, MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping = ((75,25),(0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(32, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1), activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))
          
model.compile(loss = 'mse', optimizer = 'adam')

history_object = model.fit(X_train, y_train, batch_size=32, nb_epoch=8,
                  shuffle=True, verbose=1, validation_split = 0.1)
       
model.save('model.h5')
          
from matplotlib.pyplot import plt
print(history_object.history.keys())
          
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()