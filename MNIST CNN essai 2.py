import numpy as np
import pandas as pd 
from numpy import unique, argmax
from keras.datasets.mnist import load_data 
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers import Dropout 
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.datasets import mnist 
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape, train_y.shape)
print(test_x.shape , test_y.shape)
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x .reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))
print(train_x.shape, train_y.shape)
print(test_x.shape , test_y.shape)
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0
fig = plt.figure(figsize = (10,3))
for i in range(20):
 ax= fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
 ax.imshow(np.squeeze(train_x[i]), cmap='gray')
 ax.set_title(train_y[i])
shape = train_x.shape[1:]
shape
 #CNN Model 
model = Sequential()
#adding convolutional layer 
model.add(Conv2D(32, (3,3), activation='relu', input_shape= shape))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(48, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
#compiling model 
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'] )
x=model.fit(train_x, train_y, epochs=10, batch_size = 64, verbose= 2 , validation_split = 0.1)
oss, accuracy= model.evaluate(test_x, test_y, verbose = 0)
print(f'Accuracy: {accuracy*100}')