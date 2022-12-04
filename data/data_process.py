#作用：给定一个文件夹和label，把文件夹内的视频提取出图片和音频特征，并转为npz格式，储存在特定地址
import os
import numpy as np
from moviepy.editor import *
import librosa
import cv2
import matplotlib.pyplot as plt
import soundfile

Video_Dir=r"C:\Users\DELL\Desktop\TinyM2Net\工科创数据集\视频文件\dog"
Cache_Dir=r"C:\Users\DELL\Desktop\TinyM2Net\工科创数据集\Cache"
Target_Dir=r'C:\Users\DELL\Desktop\TinyM2Net\工科创数据集\npz文件'
Save_file_name=r'dog_train'


label_list=['cat','dog','others']
label='dog'


mfccs = []
frames = []
labels = []


#根据label构造one hot encoding
one_hot_label = np.zeros(len(label_list))
for index,animal in enumerate(label_list):
    if animal in label.lower():
        one_hot_label[index] = 1

videos = os.listdir(Video_Dir)
print(videos)
for video in videos:
    id = video.split(' ')[0]
    index= video.split(' ')[1]
    start_second=int(video.split(' ')[2])
    end_second=int(video.split(' ')[3])
    location=os.path.join(Video_Dir,video)
    videoclip=VideoFileClip(location)

    #提取音频，下面要使用
    audioclip=videoclip.audio
    cache_location=os.path.join(Cache_Dir,"tmp.wav")
    audioclip.write_audiofile(cache_location)

    #提取音频特征，参考：https://blog.csdn.net/qq_23981335/article/details/115753516
    x,sr=librosa.load(cache_location,offset=start_second,duration=end_second-start_second)
    mfcc_features=librosa.feature.mfcc(x,sr=sr,n_mfcc=44)  # MFCC feature extraction from audios
    os.remove(cache_location)
    for i in range (0,end_second-start_second): #0-9
        #frame
        jpg=videoclip.get_frame(t=i+start_second)
        plt.imshow(jpg)
        plt.show()
        im=cv2.resize(jpg,(32,32))
        # plt.imshow(im)
        # plt.show()
        frames.append(np.transpose(im,(2,0,1)))  #resize img to 32x32

        step=int(mfcc_features.shape[1]/(end_second-start_second))
        mfcc_feature=mfcc_features[:,step*i:step*i+13]
        mfccs.append(np.expand_dims(mfcc_feature,axis=0))
        labels.append(one_hot_label)
        print('sasa')


location =os.path.join(Target_Dir,Save_file_name+".npz")
np.savez(location,x=mfccs,y=frames,z=labels)
        # audioclip=videoclip.audio
        # location=os.path.join(Cache_Dir,"tmp.wav")
        # audioclip.write_audiofile(location)
