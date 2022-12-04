from pytube import YouTube
import pandas as pd
from moviepy.editor import *
import os
from tqdm import *

#给定下面三个地址，下载指定的youtube视频
BASE_URL=r"https://www.youtube.com/watch?v="

RELEATIVE_PATH = r'../DataSet/'
Dataset_file_path=RELEATIVE_PATH+r"Excel/dog/dog_balanced_train_segments.xlsx"
BASE_Download_path=RELEATIVE_PATH+r"raw_video/dog"


# JPG_path=r"C:\Users\DELL\Desktop\TinyM2Net\工科创数据集\Jpg文件\dog"
# Wav_path=r"C:\Users\DELL\Desktop\TinyM2Net\工科创数据集\Wav文件\dog"
def Download(id,index,start_seconds,end_seconds):
    youtubeObject = YouTube(BASE_URL+id)
    #youtubeObject = youtubeObject.streams.get_lowest_resolution()
    path=None
    try:
        youtubeObject=youtubeObject.streams.get_lowest_resolution()
        path=youtubeObject.download(output_path=BASE_Download_path,filename_prefix=id+" "+str(index)+" "+str(start_seconds)+" "+str(end_seconds)+" ")
    except :
        print("An error has occurred for ",id)
        return path
    else:
        print(id,"is downloaded successfully")
        return path
    # youtubeObject=youtubeObject.streams.get_lowest_resolution()
    # path=youtubeObject.download(output_path=BASE_Download_path,
    #                             filename_prefix=id+" "+str(index)+" "+str(start_seconds)+" "+str(end_seconds)+" ")
    # return path

df=pd.read_excel(Dataset_file_path)



for i in trange (1,df.shape[0]):
    id=df.loc[[i],["# YTID"]].values[0][0]
    start_second=int(df.loc[[i],[" start_seconds"]].values[0][0])
    end_second=int(df.loc[[i],[" end_seconds"]].values[0][0])
    #print(id)
    #file_name=label+"_"+str(i)
    Download(str(id),i,start_second,end_second)
    # #location=os.path.join(BASE_Download_path,file_name+".mp4")
    # videoclip=VideoFileClip(location)
    # for j in range (start_second,end_second):
    #     location=os.path.join(JPG_path,file_name+str(j)+".jpg")
    #     videoclip.save_frame(location,t=j)
    #     audioclip=videoclip.audio
    #     location=os.path.join(Wav_path,file_name+str(j)+".wav")
    #     audioclip.write_audiofile(location)



