import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from base import BaseNet

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet','fmnist_LeNet','KDD_LeNet','CelebA_LeNet')
    assert net_name in implemented_networks

    net = None
    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()
    if net_name == 'fmnist_LeNet':
        net = FMNIST_LeNet()
    if net_name == 'KDD_LeNet':
        net = KDD_LeNet()   
    if net_name == 'CelebA_LeNet' :
        net = CelebA_LeNet()
        
    return net


class KDD_LeNet(BaseNet):
    def __init__(self):        
        super(KDD_LeNet, self).__init__()
        
        self.x_dim = 115
        self.encoder_h_dim= [30]
        self.rep_dim = 5
        self.decoder_h_dim= [30]
        
        self.encoder_layers = []
        self.encoder_layers.append(nn.Linear(self.x_dim, self.encoder_h_dim[0]))
        for i in range(len(self.encoder_h_dim)-1):
            self.encoder_layers.append(nn.Linear(self.encoder_h_dim[i], self.encoder_h_dim[i+1]))
        self.encoder_layers.append(nn.Linear(self.encoder_h_dim[len(self.encoder_h_dim) - 1], self.rep_dim))
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
        self.decoder_layers = []
        self.decoder_layers.append(nn.Linear(self.rep_dim, self.decoder_h_dim[0]))
        for i in range(len(self.decoder_h_dim)-1):
            self.decoder_layers.append(nn.Linear(self.decoder_h_dim[i], self.decoder_h_dim[i+1]))
        self.decoder_layers.append(nn.Linear(self.decoder_h_dim[len(self.decoder_h_dim) - 1], self.x_dim))
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        for layer_idx, layer_name in enumerate(self.encoder_layers[:-1]):
            x = self.relu(layer_name(x))
        encoded = self.encoder_layers[len(self.encoder_h_dim)](x)
        return encoded


class MNIST_LeNet(BaseNet):
    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 8 * 8, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class FMNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 8 * 8, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class CelebA_LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.W, self.C = 64, 3
        self.encoder_Conv_params= {'channels': [self.C, 64, 128, 256, 512],
                                   'kernel_size': [5, 5, 5, 5, None],
                                   'stride': [2, 2, 2, 2, None],
                                   'padding': ['SAME', 'SAME', 'SAME', 'SAME', None]}
        self.encoder_Conv_params['width'] = [self.W]
        self.rep_dim = 64
        self.decoder_Conv_params= {'channels': [512, 256, 128, 64, self.C],
                              'kernel_size': [5, 5, 5, 5, None],
                              'stride': [2, 2, 2, 1, None],
                              'padding': ['SAME', 'SAME', 'SAME', 'SAME', None]}
        self.decoder_Conv_params['width'] = [8, 16, 32, 64, 64]
        
        # encoder
        self.encoder_layers = []
        for i in range(len(self.encoder_Conv_params)-1):
            if (self.encoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.encoder_Conv_params['kernel_size'][i]-1)/2)
                out_width = int((self.encoder_Conv_params['width'][i]-self.encoder_Conv_params['kernel_size'][i]+2*padding)
                                /self.encoder_Conv_params['stride'][i]+1)
                self.encoder_Conv_params['width'].append(out_width)
            elif (self.encoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                out_width = int((self.encoder_Conv_params['width'][i]-self.encoder_Conv_params['kernel_size'][i])/self.encoder_Conv_params['stride'][i]+1)
                self.encoder_Conv_params['width'].append(out_width)
            self.encoder_layers.append(nn.Conv2d(in_channels = self.encoder_Conv_params['channels'][i],
                                                 out_channels = self.encoder_Conv_params['channels'][i+1],
                                                 kernel_size = self.encoder_Conv_params['kernel_size'][i],
                                                 stride = self.encoder_Conv_params['stride'][i],
                                                 padding = padding))
            nn.init.kaiming_normal_(self.encoder_layers[len(self.encoder_layers)-1].weight)
            nn.init.zeros_(self.encoder_layers[len(self.encoder_layers)-1].bias)
            
            self.encoder_layers.append(nn.BatchNorm2d(self.encoder_Conv_params['channels'][i+1], eps=1e-03))
            self.encoder_layers.append(nn.ReLU())
        
        self.encoder_fc_input_channels = (self.encoder_Conv_params['channels'][len(self.encoder_Conv_params) - 1]
                                          *(self.encoder_Conv_params['width'][len(self.encoder_Conv_params) - 1]**2))
        self.encoder_layers.append(nn.Linear(self.encoder_fc_input_channels, self.rep_dim))
        nn.init.kaiming_normal_(self.encoder_layers[len(self.encoder_layers)-1].weight)
        nn.init.zeros_(self.encoder_layers[len(self.encoder_layers)-1].bias)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
    def forward(self, x):
        for layer_idx, layer_name in enumerate(self.encoder_layers[:-1]):
            x = layer_name(x)
        x = x.view(x.size(0), -1)
        encoded = self.encoder_layers[len(self.encoder_layers)-1](x)
        return encoded

