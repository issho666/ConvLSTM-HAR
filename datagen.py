import cv2
from sklearn.utils import shuffle
import numpy as np
import time
import pandas as pd
import os
import collections
import subprocess
import glob
import json
# import imutils

IMAGE = (64,64)
T = 24

def video_to_frames(path_to_vid):
    frm_list = []
    
    no_of_frames, no_of_frames_extracted = 0, 0
    cap = cv2.VideoCapture(path_to_vid)
    while no_of_frames_extracted < T:
        retval, frame = cap.read()
        if retval:
            if no_of_frames%10==0:
                image = cv2.resize(frame,IMAGE)
                frm_list.append(image)
                no_of_frames_extracted+=1
                no_of_frames+=1
            else:
                no_of_frames+=1
        else:
            print("Insfitient frames")
            # print(no_of_frames, no_of_frames_extracted)
            break
    return frm_list

# list1 = video_to_frames('/home/tarun/Kinetic600/kinetics-downloader/dataset/train/juggling_balls/0NfMPvv-qLo.mp4')
# print(len(list1))

def data_generator(input_dir,input_class):
    X = []
    Y = []
    no_of_videos_skipped = 0
    videos = [vid for vid in os.listdir(input_dir) if vid.endswith('.mp4')]
    print("Starting at : ",str(len(videos)))
    for vid in videos:
        frame_list = video_to_frames(os.path.join(input_dir,vid))
        # print(frame_list)
        if len(frame_list)  <  T:
            no_of_videos_skipped+=1
            print('Removed')
            videos.remove(vid)

        else:
            X.append(frame_list)
            y = [0]*2  
            y[input_class] = 1  #one hot encoading
            Y.append(y)
            
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y

# X,Y = data_generator('/home/tarun/Kinetic600/kinetics-downloader/dataset/train/juggling_balls',1)

# print(len(X))

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

print(X_test)
print(X_test.shape)
print(y_test)
print(y_test.shape)

# res_train = dict(zip(X_train, y_train))
# res_test = np.array(zip(X_test,y_test))

# with open('train.json', 'w') as f:
#     json.dump(res_train, f)

# with open('test.json', 'w') as f:
#     json.dump(res_test, f)

# print(res_test)
# print(res_test.shape)

my_dict= {}

print(len(X_test))
print(X_test[1])
