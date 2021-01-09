from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, LSTM, Reshape, ConvLSTM2D
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
X, X_test = np.load('x_train.npz')['sequence_array'],np.load('x_test.npz')['sequence_array']
y, y_test = np.load('y_train.npz')['sequence_array'],np.load('y_test.npz')['sequence_array']
# (X, y), (X_test, y_test) = cifar10.load_data()
# X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0
# y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
#              activation='relu', data_format="channels_last"))
model.add(Conv2D(32, (3, 3), input_shape=(480, 640, 3), padding='same',
             activation='relu', data_format="channels_last"))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Reshape((239, 319, 32,1)))
# model.add(Reshape((15, 15, 32,1)))
model.add(ConvLSTM2D(32,(3,3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])

model.summary()
model.fit(X, y, validation_data=(X_test, y_test), epochs=20,batch_size=2)
#Save the weights to use for later
model.save_weights("ex1.hdf5")
#Finally print the accuracy of our model!
print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))