import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

imgs = np.load('/home/tarun/convLSTM/tarin_array.npy')

imgs = tf.squeeze(imgs,axis=-1)

for i in imgs:
    for t in i:
        z= "win"
        t = np.array(t)
        cv2.imshow(z,cv2.resize(t,(512,512)))
        k = cv2.waitKey(27)
        if k==ord('q'):
            break
