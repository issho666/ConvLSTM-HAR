import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Conv2D, TimeDistributed, LSTM, MaxPooling2D
from tensorflow.keras.models import Model
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMAGE = (64,64)
T=10

X1,Y1 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/train/juggling_balls',0)
X2,Y2 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/train/punching_bag',1)

X_train = np.concatenate((X1, X2))
y_train = np.concatenate((Y1, Y2))

X_train,y_train = shuffle(X_train,y_train)

X_test1,y_test1 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/valid/juggling_balls',0)
X_test2,y_test2 = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/valid/punching_bag',1)

X_test = np.concatenate((X_test1, X_test2))
y_test = np.concatenate((y_test1, y_test2))

X_test,y_test = shuffle(X_test,y_test)

i = Input(shape=(10,64,64,3))
x = TimeDistributed(Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer="glorot_uniform", kernel_regularizer='l2'))(i)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer="glorot_uniform", kernel_regularizer='l2'))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(64,return_sequences=False)(x)
x = Dense(2,activation='softmax')(x)

model = Model(i,x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

r = model.fit(X_train,y_train,validation_data=(X_test, y_test),batch_size=32,epochs=30)
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show() 