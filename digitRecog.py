 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import Keras packages and libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

# import utilities
import pandas as pd
from keras.utils import to_categorical



#Load dataset
train_data = pd.read_csv('./train.csv')

predictors = train_data.drop(['label'], axis=1).as_matrix()
predictors = predictors.reshape(predictors.shape[0],28,28,1)
predictors = predictors.astype('float32')

target = to_categorical(train_data.label)

# initialize CNN
classifier = Sequential()

# step1 - build Convolution Layer
classifier.add(Conv2D(32,(3,3), input_shape=(28,28,1), activation='relu'))

# step2- Add Pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Conv2D(32,(3,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))


# step3 - Flattening
classifier.add(Flatten())

# step 4  - Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=10,activation='softmax'))

#Normalize and scale input data
predictors /= 255

#compile CNN
classifier.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics=['accuracy'])

# Part 2 - fitting the CNN to images
classifier.fit(predictors,target,batch_size=32,epochs=21, validation_split=0.25)

# apply model on unseen ( test ) data
# Read test data
test_data = pd.read_csv('./test.csv')
test_data = test_data.values.astype('float32')

#  Transform test data to 4 dimnesions: n, width, height, channel
test_data = test_data.reshape(test_data.shape[0],28,28,1)

#normalize and scale test data
test_data /=255

# Prediction on test data
myArray1 = classifier.predict_classes(test_data,batch_size=32,verbose=0)

# Save predicted results in a file
import numpy as np
np.savetxt('./hdr.csv',np.dstack((np.arange(1, myArray1.size+1),myArray1))[0], fmt='%d,%d',delimiter=',', header="ImageId,Label",comments="")

