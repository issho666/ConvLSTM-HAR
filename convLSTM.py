from datagen import data_generator,video_to_frames
from sklearn.utils import shuffle

import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model


pf = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(pf[0],True)

# X1,Y1 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/train/juggling_balls',0)
# X2,Y2 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/train/punching_bag',1)

# X_train = np.concatenate((X1, X2))
# y_train = np.concatenate((Y1, Y2))

# X_train,y_train = shuffle(X_train,y_train)

X_test1,y_test1 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/valid/juggling_balls',0)
X_test2,y_test2 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/valid/punching_bag',1)

X_test = np.concatenate((X_test1, X_test2))
y_test = np.concatenate((y_test1, y_test2))

X_test,y_test = shuffle(X_test,y_test)


T = 24
IMAGE = (64,64)
i = Input(shape=(T, 64, 64, 3))
x = ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same',return_sequences = True)(i)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(2,activation='softmax')(x)

model = Model(i,x)
model.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

r = model.fit(X_test,y_test,batch_size=32,epochs=10)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('test.h5')