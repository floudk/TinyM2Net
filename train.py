# %% md

# import packages

# %%

# Numerical Operations
import math
import numpy as np

# For Progress Bar
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os

import datetime
class TinyM2NetDataset(Dataset):
    '''
    x: audio mfcc vector   44x13x1.
    y: image vector        32x32x3
    y: Targets:(cat,dog,duck,rabbit), if none, do prediction.
    '''

    def __init__(self, x, y, z=None):
        if y is None:
            self.z = z
        else:
            self.z = torch.FloatTensor(z)
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        if self.z is None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx], self.z[idx]

    def __len__(self):
        return len(self.x)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.outlayer = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.outlayer(out)
        return out


class myConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dense_dim, bn_dim):
        super(myConv2d, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=1),
            nn.BatchNorm2d(bn_dim),
            nn.ReLU()
        )
        self.spconv2d1 = nn.Sequential(
            SeparableConv2d(output_channels, 32, kernel_size),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2)
        )
        self.spconv2d2 = nn.Sequential(
            SeparableConv2d(32, output_channels, kernel_size),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2)
        )
        self.outlayer = nn.Sequential(
            nn.Linear(dense_dim, output_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        #         print(x.shape)
        out = self.conv2d(x)
        #         print(out.shape)
        out = self.spconv2d1(out)
        #         print(out.shape)
        out = self.spconv2d2(out)
        #         print(out.shape)
        out = torch.flatten(out, start_dim=1)
        #         print(out.shape)
        out = self.outlayer(out)
        #         print(out.shape)
        return out


class Tiny2Net(nn.Module):
    def __init__(self, labels, device):
        super(Tiny2Net, self).__init__()
        #         self.args = args
        self.videoNet = myConv2d(3, 64, (3, 3), 4096, 64)  # (3,64,(3,3),4096,32)
        self.audioNet = myConv2d(1, 64, (3, 3), 2112, 64)  # (1,64,(3,3),2112,44)
        self.layer1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, labels),
            nn.Softmax(dim=-1)
        )
        self.device=device
    def forward(self, x, y):
        """
        input x   MFCC Vector     size:  44x13x1
        input y   Image Vector   size: 32x32x3
        """
        x_noise, y_noise = torch.rand_like(x).to(device), torch.rand_like(y).to(device)
        x = self.audioNet(x+x_noise.detach())
        y = self.videoNet(y+y_noise.detach())
        z = torch.cat((x, y), 1)
        #         print("z:",z.shape)
        z = self.layer1(z)
        #         print("z:",z.shape)
        z = self.layer2(z)
        #print("z:", z.shape)
        return z#nn.functional.softmax(z, dim=-1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
        'seed': 42,
        'valid_ratio': 0.2,
        'n_epochs': 1000,
        'batch_size': 1024,
        'learning_rate': 5e-3,
        'early_stop': 50,
        'save_path': './models/',  # model will be saved here.
        'data_path': './DataSet/npz/',
        'data_file': ["cat_eval", "cat_train",
                      "dog_eval", "dog_train",
                      "other_eval", "other_train"],
    }


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(x, y, z, valid_ratio, seed, batch_size):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(x))
    train_set_size = len(x) - valid_set_size
    data_index = np.arange(len(x))
    train_index, valid_index = random_split(data_index, [train_set_size, valid_set_size],
                                            generator=torch.Generator().manual_seed(seed))
    train_index, valid_index = np.array(train_index), np.array(valid_index)

    return x[train_index], y[train_index], z[train_index], x[valid_index], y[valid_index], z[valid_index]


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x, y)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def loadData(path, dataList):
    def getName(base, file):
        return base + file + '.npz'

    dataset = {}
    for item in dataList:
        dataset[item] = np.load(getName(path, item))

    return dataset


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    criterion = nn.CrossEntropyLoss()

    #     optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975, last_epoch=-1, verbose=True)
    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode.
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y, z in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.

            x, y, z = x.to(device), y.to(device), z.to(device)
            pred = model(x, y)
            #             print("train: ",pred)
            target = z.argmax(dim=1, keepdim=False)
            loss = criterion(pred, target)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(torch.mean(loss).detach().item())
            train_correct = torch.argmax(pred, dim=1) == torch.argmax(z, dim=1)
            train_accuracy = torch.mean(train_correct.float())
            writer.add_scalar('Acc/train', train_accuracy, step)
            writer.add_scalar('Loss/train', torch.mean(loss), step)
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            train_pbar.set_postfix({'Acc': train_accuracy.detach().item()})
        scheduler.step()
        mean_train_loss = sum(loss_record) / len(loss_record)
        #writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set the model to evaluation mode.
        loss_record = []
        for x, y, z in valid_loader:
            x, y, z = x.to(device), y.to(device), z.to(device)
            with torch.no_grad():
                pred = model(x, y)
                target = z.argmax(dim=1, keepdim=False)
                loss = criterion(pred, target)
            val_correct = torch.argmax(pred, dim=1) == torch.argmax(z, dim=1)
            val_accuracy = torch.mean(val_correct.float())
            writer.add_scalar('Acc/valid', val_accuracy, step)
            writer.add_scalar('Loss/valid', torch.mean(loss), step)
            loss_record.append(torch.mean(loss).item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        #writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']+str(best_loss))  # Save the best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        #if early_stop_count >= config['early_stop']:
        #    print('\nModel is not improving, so we halt the training session.')
        #    return

if __name__ == '__main__':
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    seed = config['seed']
    config['save_path'] = config['save_path'] + f'seed_{seed}_{t0}_entropy_standardnoise_.pth'


    same_seed(config['seed'])
    # load data
    dataset = loadData(config['data_path'], config['data_file'])
    x = None
    y = None
    z = None
    for k, v in dataset.items():
        if x is None:
            x = v["x"]
            y = v["y"]
            z = v["z"]
        else:
            x = np.concatenate((x, v["x"]), axis=0)
            y = np.concatenate((y, v["y"]), axis=0)
            z = np.concatenate((z, v["z"]), axis=0)

    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(len(x))

    # %%

    train_x, train_y, train_z, valid_x, valid_y, valid_z = train_valid_split(x, y, z, config['valid_ratio'],
                                                                             config['seed'],
                                                                             config['batch_size'])
    print(len(train_x))
    print(len(valid_x))

    # %%

    train_dataset, valid_dataset = TinyM2NetDataset(train_x, train_y, train_z), TinyM2NetDataset(valid_x, valid_y,
                                                                                                 valid_z)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    print(config)
    print(device)

    model = Tiny2Net(z.shape[-1], device).to(device)

    trainer(train_loader, valid_loader, model, config, device)