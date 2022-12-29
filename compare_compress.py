# Numerical Operations
import math
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # self.layer1 = nn.Sequential(
        #     depthwise_separable_conv(output_channels,3,32),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.2)
        # )
        # self.layer2 = nn.Sequential(
        #     depthwise_separable_conv(32,3,output_channels),
        #     nn.MaxPool2d((2,2)),
        #     nn.Dropout(0.2)
        # )
        #################################################
        
        
        ######## use normal conv #########################
        self.layer1 = nn.Sequential(
            nn.Conv2d(output_channels,32,kernel_size,padding=1),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,output_channels,kernel_size,padding=1),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.2)
        )
        ################################################
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


def test_model(model, valid_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode.
    loss_record = []
    acc_record = []
    for x, y, z in valid_loader:
        x, y, z = x.to(device), y.to(device), z.to(device)
        with torch.no_grad():
            pred = model(x, y)
            target = z.argmax(dim=1, keepdim=False)
            loss = criterion(pred, target)
        val_correct = torch.argmax(pred, dim=1) == torch.argmax(z, dim=1)
        val_accuracy = torch.mean(val_correct.float())
        acc_record.append(val_accuracy)
        loss_record.append(torch.mean(loss).item())
    
    print("the average accuracy: ", sum(acc_record)/len(acc_record))


def run():
    data = loadData(config['data_path'],config['data_file'])
    # 按照种类加载
    X = [None,None,None]
    Y = [None,None,None]
    Z = [None,None,None]
    label=2
    for k,v in data.items():
        if 'cat' in k:
            label=0
        elif 'dog' in k:
            label=1
        else:
            label=2
            
        if X[label] is None:
            X[label] = v["x"]
            Y[label] = v["y"]
            Z[label] = v["z"]
        else:
            X[label] = np.concatenate((X[label], v["x"]), axis=0)
            Y[label] = np.concatenate((Y[label], v["y"]), axis=0)
            Z[label] = np.concatenate((Z[label], v["z"]), axis=0)
    # print(X,Y,Z)
    # 按照种类split
    TRAIN_X =  [None,None,None]
    TRAIN_Y = [None,None,None]
    TRAIN_Z = [None,None,None]
    VAL_X =  [None,None,None]
    VAL_Y =  [None,None,None]
    VAL_Z =  [None,None,None]
    for i in [0,1,2]:
        _,_,_,VAL_X[i],VAL_Y[i],VAL_Z[i]=\
                                train_valid_split(X[i],Y[i],Z[i],config['valid_ratio'],config['seed'])
        
    # 合并TRAIN, VAL
    #train_x,train_y,train_z = TRAIN_X[0],TRAIN_Y[0],TRAIN_Z[0]
    val_x,val_y,val_z = VAL_X[0],VAL_Y[0],VAL_Z[0]
    for i in [1,2]:
        # train_x = np.concatenate((train_x,TRAIN_X[i]),axis=0)
        # train_y = np.concatenate((train_y,TRAIN_Y[i]),axis=0)
        # train_z = np.concatenate((train_z,TRAIN_Z[i]),axis=0)
        val_x = np.concatenate((val_x,VAL_X[i]),axis=0)
        val_y = np.concatenate((val_y,VAL_Y[i]),axis=0)
        val_z = np.concatenate((val_z,VAL_Z[i]),axis=0)

    # print(train_x.shape)
    # print(train_y.shape)
    # print(train_z.shape)
    print(val_x.shape)
    print(val_y.shape)
    print(val_z.shape)
    valid_dataset =  TinyM2NetDataset(val_x,val_y,val_z)

    # Pytorch data loader loads pytorch dataset into batches.
    #train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    model = Tiny2Net(val_z.shape[-1], device).to(device)
    model_path = "./models"
    
    basepath = "seed_42_1220_113359_base_0.6479765892028808.pth"
    normalpath = "seed_42_1223_032058_normconv2d_0.7915669977664948.pth"
    videopath = "seed_42_1220_104432_vedio_0.6901510119438171.pth"
    audiopath = "seed_42_1220_102443_audio_0.8027129471302032.pth"
    path = os.path.join(model_path, normalpath) #选择不同的模型进行测试，选择不同模型时，需要对模型结构进行修改
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    Quantization = False #是否量化
    if Quantization:
        torch.backends.quantized.engine = 'qnnpack'
        model_int8 = torch.quantization.quantize_dynamic(model,  # the original model
        {torch.nn.Linear,torch.nn.Conv2d,nn.MaxPool2d},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
        print("begin to test ..................")
        starttime = time.time()
        test_model(model_int8, valid_loader)
        endtime = time.time()
        print("quantization model spend time to test: ", endtime-starttime)
        print("test end .......................")
    
    else:
        print("begin to test ..................")
        starttime = time.time()
        test_model(model, valid_loader)
        endtime = time.time()
        print("normal model spend time to test: ", endtime-starttime)
        print("test end .......................")

if __name__=="__main__":
    run()
   