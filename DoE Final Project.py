#Import libraries
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras import optimizers
from keras.datasets import cifar10

#Load data
(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size= 0.25)
X_train=X_train[:20000,:,:,:]
y_train=y_train[:20000,:]
X_val=X_val[:1000,:,:,:]
y_val=y_val[:1000,:]

labels=dict()
labels["0"]="airplane"
labels["1"]="automobile"
labels["2"]="bird"
labels["3"]="cat"
labels["4"]="deer"
labels["5"]="dog"
labels["6"]="frog"
labels["7"]="horse"
labels["8"]="ship"
labels["9"]="truck"

index=np.random.choice(X_train.shape[0], 4, False)
# Lets see the dataset!
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[index[0]], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[index[1]], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[index[2]], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[index[3]], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


X_train, X_val=prep_pixels(X_train, X_val)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

#Range of the hyperparameters
#Kernel (filter) size Conv2D: 1, 3, 5, 7, 11 (integer)
#Number of convolutional layers = 1 - 4 (integer)
#Number of filter conv2D = 16, 32, 64, 128, 264 (integer)
#Activation = relu, elu, tanh, sigmoid (categorical)
#Number of neurons = 50 - 300 (integer)
#Learning rate= 0.0001 - 1 (continuous)
#Number of epochs: 1 - 40
#Batch size = 1-100 (integer)
#momentum = 0.5 - 0.99 (continuous)

#Hypperparameters values
kernel_size_Conv2D = 5
number_of_convolutional_layers = 1
number_of_filter_conv2D = 64
activation = 'sigmoid'
number_of_neurons = 300
learning_rate = 0.001
number_of_epochs = 10
batch_size = 10
momentum = 0.8

#Train the convolutional neural network
model = Sequential()

model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((3, 3)))

if number_of_convolutional_layers >= 2:
        model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((3, 3)))

if number_of_convolutional_layers >= 3:
        model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((3, 3)))

if number_of_convolutional_layers >= 4:
        model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same'))

model.add(Flatten())
model.add(Dense(number_of_neurons, activation = activation, kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

opt = SGD(lr = learning_rate, momentum = momentum)
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs = number_of_epochs, batch_size = batch_size, validation_data = (X_val, y_val), verbose = 1)