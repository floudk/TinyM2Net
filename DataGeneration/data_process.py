#作用：给定一个文件夹和label，把文件夹内的视频提取出图片和音频特征，并转为npz格式，储存在特定地址
import os
import numpy as np
from moviepy.editor import *
import librosa
import cv2
import matplotlib.pyplot as plt
from tqdm import *

Video_Dir=r"/home/junxianh/projects/Yuzhen/video文件/dog/eval"
Cache_Dir=r"/home/junxianh/projects/Yuzhen/cache"
Target_Dir=r'/home/junxianh/projects/Yuzhen/npz_new'
Save_file_name=r'dog_eval'


label_list=['cat','dog','other']
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
#print(videos)
for video in tqdm(videos):
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
    sample=np.arange(start_second,end_second,0.5).tolist()
    for i, sample_point in enumerate(sample): #0-9
        #frame
        jpg=videoclip.get_frame(t=sample_point)
        # plt.imshow(jpg)
        # plt.show()
        im=cv2.resize(jpg,(64,64))
        # plt.imshow(im)
        # plt.show()
        frames.append(np.transpose(im,(2,0,1)))  #resize img to 32x32

        step=int(mfcc_features.shape[1]/(len(sample)))
        mfcc_feature=mfcc_features[:,step*i:step*i+13]
        assert mfcc_feature.shape[0]==44
        assert mfcc_feature.shape[1]==13
        mfccs.append(np.expand_dims(mfcc_feature,axis=0))
        labels.append(one_hot_label)
        #print('sasa')


location =os.path.join(Target_Dir,Save_file_name+".npz")
np.savez(location,x=mfccs,y=frames,z=labels)
        # audioclip=videoclip.audio
        # location=os.path.join(Cache_Dir,"tmp.wav")
        # audioclip.write_audiofile(location)
