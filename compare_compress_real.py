# Numerical Operations
import math
import numpy as np
from CameraAudioRead.AudioRead import read_audio
from CameraAudioRead.CameraRead import read_camera
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
import glob
import librosa
import cv2

config = {
    'seed': 42,   
    'valid_ratio': 0.3,
    'n_epochs': 10,        
    'batch_size': 16, 
    'learning_rate': 5e-5,              
    'early_stop': 3,    
    'save_path': './models/model.ckpt',  # model will be saved here.
    'data_path': './DataSet/npz_new/',
    'data_file': ["cat_eval","cat_train",
                  "dog_eval", "dog_train",
                  "other_eval","other_train"],
}
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
same_seed(config['seed'])
if not os.path.isdir('./models'):
    os.mkdir('./models')

class TinyM2NetDataset(Dataset):
    '''
    x: audio mfcc vector   44x13x1.
    y: image vector        64x64x3
    y: Targets:(cat,dog,duck,rabbit), if none, do prediction.
    '''
    def __init__(self, x,y,z=None):
        if y is None:
            self.z = z
        else:
            self.z = torch.FloatTensor(z)
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __getitem__(self, idx):
        if self.z is None:
            return self.x[idx],self.y[idx]
        else:
            return self.x[idx], self.y[idx], self.z[idx]
    def __len__(self):
        return len(self.x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,groups=in_channels,padding=1)
        self.pointwise = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=1)
        self.outlayer = nn.ReLU()
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.outlayer(out)
        return out
    
    
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout): 
        super(depthwise_separable_conv, self).__init__() 
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 
        self.relu = nn.ReLU()
    def forward(self, x): 
        out = self.depthwise(x) 
        out = self.pointwise(out)
        out = self.relu(out)
        return out

    
class myConv2d(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,dense_dim,bn_dim):
        super(myConv2d, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size,padding=1),
            nn.BatchNorm2d(bn_dim),
            nn.ReLU()
        )
        ########## use Speratle conv ##################
        self.layer1 = nn.Sequential(
            depthwise_separable_conv(output_channels,3,32),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            depthwise_separable_conv(32,3,output_channels),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2)
        )
        #################################################
        
        
        ######### use normal conv #########################
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(output_channels,32,kernel_size,padding=1),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.2)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(32,output_channels,kernel_size,padding=1),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.2)
        # )
        #################################################
        self.outlayer = nn.Sequential(
            nn.Linear(dense_dim,output_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self,x):
        out = self.conv2d(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = torch.flatten(out,start_dim=1)
        out = self.outlayer(out)
        return out


class Tiny2Net(nn.Module):
    def __init__(self, labels,device):
        super(Tiny2Net, self).__init__()
        
        self.videoNet = myConv2d(3,64,(3,3),16384,64)  #(3,64,(3,3),4096,32)
        self.audioNet = myConv2d(1,64,(3,3),2112,64) #(1,64,(3,3),2112,44)
        
        self.layer1 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(64,labels),
            nn.Softmax(dim=-1)
        )
        self.device=device
        
    def forward(self,x,y):
        """
        input x   MFCC Vector     size:  44x13x1
        input y   Image Vector   size: 32x32x3
        """
        x_noise, y_noise = torch.rand_like(x).to(device), torch.rand_like(y).to(device)
        x = self.audioNet(x+x_noise.detach()) #audio
        y = self.videoNet(y+y_noise.detach())
        
        z = torch.cat((x,y),1)#torch.cat((x,y),1)
        z = self.layer1(z)
        z = self.layer2(z)
        return z

def loadData(path,dataList):
    def getName(base,file):
        return base+file+'.npz'
    dataset={}
    for item in dataList:
        dataset[item]=np.load(getName(path,item))
        
    return dataset

def train_valid_split(x,y,z, valid_ratio,seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(x)) 
    train_set_size = len(x) - valid_set_size
    data_index = np.arange(len(x))
    train_index, valid_index = random_split(data_index, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    train_index, valid_index = np.array(train_index), np.array(valid_index)

    return x[train_index],y[train_index],z[train_index],x[valid_index],y[valid_index],z[valid_index]





def test_real(model,device):
    """
    直接使用摄像头和麦克风对环境进行识别
    """
    reslutfile = "./LOG/result_"+str(int(time.time()))+".log"
    model.eval() 
    animal_dict = {0:"cat",1:"dog",2:"other animal"}
    while(True):
        
        resultfd = open(reslutfile,'a+')
        camera_frames = read_camera(resize_h=64, resize_w=64, frames_number=1)
        audio_frames = read_audio(record_second=1,frames_number=1)
        # print(camera_frames[0])
        # print(audio_frames[0])
        y = torch.from_numpy(camera_frames[0]).unsqueeze(0).to(device)
        x = torch.from_numpy(audio_frames[0]).unsqueeze(0).to(device)
        # print(x.shape)
        # print(y.shape)
        with torch.no_grad():
            pred = model(x, y)
            target_list = pred.cpu().numpy().tolist()[0]
            target = target_list.index(max(target_list))
            print("predict list {} the prediction is {} ,corresponding animal is {}\n".format(target_list,target,animal_dict[target]), file=resultfd)
            print("predict list {} the prediction is {} ,corresponding animal is {}".format(target_list,target,animal_dict[target]))

        resultfd.close()
        time.sleep(1)

def test_1(model, device, cats_rgb_path, cats_audio_path, dogs_rgb_path, dogs_audio_path):
    '''
    对输入的模型，进行测试，对应的猫和狗的rgb和audio的数据路径
    '''
   
    cats_rgbs = glob.glob(cats_rgb_path+'/*')
  
    cat_right = 0
    for cat_rgb in cats_rgbs:
        jpg = cv2.imread(cat_rgb)
        im=cv2.resize(jpg,(64,64))
        frame = np.transpose(im, (2,0,1))
        frame = frame.astype(np.float32)

        basenames = os.path.basename(cat_rgb).split('_')
        sample_point = int(basenames[-1].split('.')[0])

        sound_location = os.path.join(cats_audio_path,basenames[0]+"_"+basenames[1]+"_"+basenames[2]+".wav")
        x,sr=librosa.load(sound_location)
        mfcc_features=librosa.feature.mfcc(x,sr=sr,n_mfcc=44)  # MFCC feature extraction from audios

        step=int(mfcc_features.shape[1]/10)
        mfcc_feature=mfcc_features[:,step*sample_point:step*sample_point+13]
        assert mfcc_feature.shape[0]==44
        assert mfcc_feature.shape[1]==13
        mfcc_feature = np.expand_dims(mfcc_feature,axis=0)

        y = torch.from_numpy(frame).unsqueeze(0).to(device)
        x = torch.from_numpy(mfcc_feature).unsqueeze(0).to(device)
        pred = model(x, y)
        target_list = pred.cpu().detach().numpy().tolist()[0]
        target = target_list.index(max(target_list))
        if target == 0:
            cat_right += 1
        #print("cat complete one, right is {}, answer is {}".format(0, target) )
   


    dogs_rgbs = glob.glob(dogs_rgb_path+'/*')
    dog_right = 0
    for dog_rgb in dogs_rgbs:
        jpg = cv2.imread(dog_rgb)
        im=cv2.resize(jpg,(64,64))
        frame = np.transpose(im, (2,0,1))
        frame = frame.astype(np.float32)

        basenames = os.path.basename(dog_rgb).split('_')
        sample_point = int(basenames[-1].split('.')[0])

        sound_location = os.path.join(dogs_audio_path,basenames[0]+"_"+basenames[1]+"_"+basenames[2]+".wav")
        x,sr=librosa.load(sound_location)
        mfcc_features=librosa.feature.mfcc(x,sr=sr,n_mfcc=44)  # MFCC feature extraction from audios

        step=int(mfcc_features.shape[1]/10)
        mfcc_feature=mfcc_features[:,step*sample_point:step*sample_point+13]
        assert mfcc_feature.shape[0]==44
        assert mfcc_feature.shape[1]==13
        mfcc_feature = np.expand_dims(mfcc_feature,axis=0)

        y = torch.from_numpy(frame).unsqueeze(0).to(device)
        x = torch.from_numpy(mfcc_feature).unsqueeze(0).to(device)
        pred = model(x, y)
        target_list = pred.cpu().detach().numpy().tolist()[0]
        target = target_list.index(max(target_list))
        if target == 1:
            dog_right += 1
        #print("dog complete one, right is {}, answer is {}".format(1, target) )

    print("cat number: {}, cat right: {}, cat right percentage: {}".format(len(cats_rgbs), cat_right,cat_right/len(cats_rgbs)))
    print("dog number: {}, dog right: {}, dog right percentage: {}".format(len(dogs_rgbs), dog_right,dog_right/len(dogs_rgbs)))



def tests(model, device):
    """
    测试四种组合
    """
    cats_rgb_path = "./TestData/rgbs/cats"
    cats_audio_path = "./TestData/audios/cats"
    dogs_rgb_path = "./TestData/rgbs/dogs"
    dogs_audio_path = "./TestData/audios/dogs"

    cats_rgb_path_camera = "./TestData/camera_rgbs/cats"
    cats_audio_path_mic = "./TestData/mic_audios/cats"
    dogs_rgb_path_camera = "./TestData/camera_rgbs/dogs"
    dogs_audio_path_mic = "./TestData/mic_audios/dogs"


    #1 audio直接读取+image 直接读取
    test_1(model,device, cats_rgb_path, cats_audio_path, dogs_rgb_path, dogs_audio_path)

    #2 audio麦克风+image直接读取
    test_1(model,device, cats_rgb_path, cats_audio_path_mic, dogs_rgb_path, dogs_audio_path_mic)

    # audio直接读取+image摄像头
    test_1(model,device, cats_rgb_path_camera, cats_audio_path, dogs_rgb_path_camera, dogs_audio_path)

    # audio麦克风+image摄像头
    test_1(model,device, cats_rgb_path_camera, cats_audio_path_mic, dogs_rgb_path_camera, dogs_audio_path_mic)




if __name__=="__main__":

    device ='cpu'# torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Tiny2Net(3, device).to(device)
    model_path = "./models"
    
    basepath = "seed_42_1220_113359_base_0.6479765892028808.pth"
    path = os.path.join(model_path, basepath)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    Quantization = True #是否进行模型量化
    if Quantization:
        torch.backends.quantized.engine = 'qnnpack'
        model_int8 = torch.quantization.quantize_dynamic(model,  # the original model
        {torch.nn.Linear,torch.nn.Conv2d,nn.MaxPool2d},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
        print("quantization begin to test ..................")
        tests(model_int8, device) #改为test_real(model_int8, device)可对环境进行测试
        
    
    else:
        print("begin to test ..................")
        tests(model,device)#改为test_real(model, device)可对环境进行测试
        
