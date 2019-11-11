# **Behavioral Cloning** 

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


#### 1. An appropriate model architecture has been employed

My model consists of three convolution neural network with 5x5 filter sizes and 2 with 3X3 filter size and depths between 24 and 64 (model.py lines 71-81) 

The model includes 5 RELU layers to introduce nonlinearity (code line 71-81), and the data is normalized in the model using a Keras lambda layer (code line 69). 

and Flatten included.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers and MaxPooling in order to reduce overfitting (model.py lines 71-91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 20-40). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road
counter-clock and vice; specifical drive on turning

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was from Nvidia 'End to End Learning for Self-Driving Cars'


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with adding more Maxpooling and dropout rate

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 70-94) consisted of a convolution neural network with the following layers and layer sizes 
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 60, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 60, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 160, 24)       1824      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 80, 24)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 40, 36)         21636     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 20, 36)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 2, 10, 48)         43248     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1, 5, 48)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 5, 64)          27712     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 5, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 5, 64)          36928     
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 5, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 320)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               32100     
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_4 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_5 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 169,019
Trainable params: 169,019
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles

After the collection process, I had 53906 number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by validation and test loss become up and down.
