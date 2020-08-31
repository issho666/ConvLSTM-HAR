import cv2
import numpy as np
import pandas as pd
import os
import collections
import subprocess
import glob
from math import floor, ceil
from sklearn.utils import shuffle

IMAGE = (64,64)
T = 10

IMAGE = (64,64)
T = 10
def video_to_frames(path_to_vid):
   
    cap = cv2.VideoCapture(path_to_vid)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frm_list = []
    counter = 0
    valid_from = ceil(0.15*n_frames)
    valid_to = floor(0.1*n_frames)
    num_frames = n_frames-valid_from-valid_to
    frame_to_count = floor(num_frames/T)
    no_of_frames_extracted = 0
    while no_of_frames_extracted < T:
        ret, frame = cap.read()
        if ret==True:
            if counter>=valid_from and (counter-valid_from)//frame_to_count==0:
                img = cv2.resize(frame,IMAGE)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                np.expand_dims(gray,axis=1)
                frm_list.append(gray)
                no_of_frames_extracted+=1
                counter+=1
            else:
                counter+=1
        else:
            break
            
    return frm_list

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
    return np.expand_dims(X,-1),Y
            
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y
