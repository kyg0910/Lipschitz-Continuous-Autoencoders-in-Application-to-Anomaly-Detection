import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseNet

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('MNIST_LCAE', 'FMNIST_LCAE', 'KDD_LCAE', 'CelebA_LCAE')
    assert net_name in implemented_networks

    net = None
    if net_name == 'MNIST_LCAE':
        net = MNIST_LCAE()
    if net_name == 'FMNIST_LCAE':
        net = FMNIST_LCAE()
    if net_name == 'KDD_LCAE' :
        net = KDD_LCAE()
    if net_name == 'CelebA_LCAE' :
        net = CelebA_LCAE()
    
    return net

def build_decoder_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('MNIST_LCAE_Decoder','CelebA_LCAE_Decoder','FMNIST_LCAE_Decoder')
    assert net_name in implemented_networks

    net = None
    if net_name == 'MNIST_LCAE_Decoder':
        net = MNIST_LCAE_Decoder()
    if net_name == 'FMNIST_LCAE_Decoder':
        net = FMNIST_LCAE_Decoder()
    if net_name == 'CelebA_LCAE_Decoder' :
        net = CelebA_LCAE_Decoder()
    
    return net

class KDD_LCAE(BaseNet):
    def __init__(self):        
        super(KDD_LCAE, self).__init__()
        
        self.x_dim = 115
        self.encoder_h_dim= [30]
        self.z_dim = 5
        self.decoder_h_dim= [30]
        
        self.encoder_layers = []
        self.encoder_layers.append(nn.Linear(self.x_dim, self.encoder_h_dim[0]))
        self.encoder_layers.append(nn.ReLU())
        for i in range(len(self.encoder_h_dim)-1):
            self.encoder_layers.append(nn.Linear(self.encoder_h_dim[i], self.encoder_h_dim[i+1]))
            self.encoder_layers.append(nn.ReLu())
        self.encoder_layers.append(nn.Linear(self.encoder_h_dim[len(self.encoder_h_dim) - 1], self.z_dim))
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
        self.decoder_layers = []
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_h_dim[0]))
        self.decoder_layers.append(nn.ReLU())
        for i in range(len(self.decoder_h_dim)-1):
            self.decoder_layers.append(nn.Linear(self.decoder_h_dim[i], self.decoder_h_dim[i+1]))
            self.decoder_layers.append(nn.ReLu())
        self.decoder_layers.append(nn.Linear(self.decoder_h_dim[len(self.decoder_h_dim) - 1], self.x_dim))
        self.decoder_layers.append(nn.Tanh())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def forward(self, x):
        for layer_idx, layer_name in enumerate(self.encoder_layers[:-1]):
            x = layer_name(x)
            
        encoded = self.encoder_layers[-1](x)
        reconstructed = encoded
        for layer_idx, layer_name in enumerate(self.decoder_layers):
            reconstructed = layer_name(reconstructed)

        return encoded, reconstructed

class MNIST_LCAE(BaseNet):
    def __init__(self):
        super().__init__()
        
        self.W, self.C = 32, 1
        self.encoder_Conv_params= {'channels': [self.C, 64, 128, 256],
                                   'kernel_size': [5, 5, 5, None],
                                   'stride': [2, 2, 2, None],
                                   'padding': ['SAME', 'SAME', 'SAME', None]}
        self.encoder_Conv_params['width'] = [self.W]
        self.z_dim = 8
        self.decoder_Conv_params= {'channels': [256, 128, 64, self.C],
                              'kernel_size': [5, 5, 5, None],
                              'stride': [2, 2, 2, None],
                              'padding': ['SAME', 'SAME', 'SAME',  None]}
        self.decoder_Conv_params['width'] = [4, 8, 16, 32]
        
        # encoder
        self.encoder_layers = []   
        for i in range(len(self.encoder_Conv_params['channels'])-1):
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
            self.encoder_layers.append(nn.BatchNorm2d(self.encoder_Conv_params['channels'][i+1], eps=1e-03))
            self.encoder_layers.append(nn.ReLU())
        
        self.encoder_fc_input_channels = (self.encoder_Conv_params['channels'][len(self.encoder_Conv_params['channels']) - 1]
                                          *(self.encoder_Conv_params['width'][len(self.encoder_Conv_params['channels']) - 1]**2))
        self.encoder_layers.append(nn.Linear(self.encoder_fc_input_channels, self.z_dim))
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
        # decoder
        self.decoder_layers = []
        self.decoder_fc_output_channels = (self.decoder_Conv_params['channels'][0]
                                          *(self.decoder_Conv_params['width'][0]**2))
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_fc_output_channels))
        for i in range(len(self.decoder_Conv_params['channels'])-1):
            if (self.decoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.decoder_Conv_params['kernel_size'][i]-1)/2)
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            elif (self.decoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels = self.decoder_Conv_params['channels'][i],
                                                          out_channels = self.decoder_Conv_params['channels'][i+1],
                                                          kernel_size = self.decoder_Conv_params['kernel_size'][i],
                                                          stride = self.decoder_Conv_params['stride'][i],
                                                          padding = padding,
                                                          output_padding = output_padding))
            self.decoder_layers.append(nn.BatchNorm2d(self.decoder_Conv_params['channels'][i+1], eps=1e-03))
            if (i < (len(self.decoder_Conv_params['channels'])-2)):
                self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        
    def forward(self, x):
        for layer_idx, layer_name in enumerate(self.encoder_layers[:-1]):
            x = layer_name(x)
        x = x.view(x.size(0), -1)
        encoded = self.encoder_layers[len(self.encoder_layers)-1](x)
        
        x = self.decoder_layers[0](encoded)
        
        x = x.view(x.size(0), self.decoder_Conv_params['channels'][0],
                         self.decoder_Conv_params['width'][0], self.decoder_Conv_params['width'][0])
        for layer_idx, layer_name in enumerate(self.decoder_layers[1:]):
            x = layer_name(x)
        reconstructed = x
        return encoded, reconstructed
    
class MNIST_LCAE_Decoder(BaseNet):
    def __init__(self):
        super().__init__()

        self.W, self.C = 32, 1
        self.z_dim = 8
        self.decoder_Conv_params= {'channels': [256, 128, 64, self.C],
                              'kernel_size': [5, 5, 5, None],
                              'stride': [2, 2, 2, None],
                              'padding': ['SAME', 'SAME', 'SAME',  None]}
        self.decoder_Conv_params['width'] = [4, 8, 16, 32]
        
        # decoder
        self.decoder_layers = []
        self.decoder_fc_output_channels = (self.decoder_Conv_params['channels'][0]
                                          *(self.decoder_Conv_params['width'][0]**2))
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_fc_output_channels))
        for i in range(len(self.decoder_Conv_params['channels'])-1):
            if (self.decoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.decoder_Conv_params['kernel_size'][i]-1)/2)
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            elif (self.decoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels = self.decoder_Conv_params['channels'][i],
                                                          out_channels = self.decoder_Conv_params['channels'][i+1],
                                                          kernel_size = self.decoder_Conv_params['kernel_size'][i],
                                                          stride = self.decoder_Conv_params['stride'][i],
                                                          padding = padding,
                                                          output_padding = output_padding))
            self.decoder_layers.append(nn.BatchNorm2d(self.decoder_Conv_params['channels'][i+1], eps=1e-03))
            if (i < (len(self.decoder_Conv_params['channels'])-2)):
                self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        
    def forward(self, x):
        x = self.decoder_layers[0](x)
        x = x.view(x.size(0), self.decoder_Conv_params['channels'][0],
                         self.decoder_Conv_params['width'][0], self.decoder_Conv_params['width'][0])
        for layer_idx, layer_name in enumerate(self.decoder_layers[1:]):
            x = layer_name(x)
        reconstructed = x
        return reconstructed

class FMNIST_LCAE(BaseNet):
    def __init__(self):
        super().__init__()
        
        self.W, self.C = 32, 1
        self.encoder_Conv_params= {'channels': [self.C, 64, 128, 256],
                                   'kernel_size': [5, 5, 5, None],
                                   'stride': [2, 2, 2, None],
                                   'padding': ['SAME', 'SAME', 'SAME', None]}
        self.encoder_Conv_params['width'] = [self.W]
        self.z_dim = 8
        self.decoder_Conv_params= {'channels': [256, 128, 64, self.C],
                              'kernel_size': [5, 5, 5, None],
                              'stride': [2, 2, 2, None],
                              'padding': ['SAME', 'SAME', 'SAME',  None]}
        self.decoder_Conv_params['width'] = [4, 8, 16, 32]
        
        # encoder
        self.encoder_layers = []   
        for i in range(len(self.encoder_Conv_params['channels'])-1):
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
            self.encoder_layers.append(nn.BatchNorm2d(self.encoder_Conv_params['channels'][i+1], eps=1e-03))
            self.encoder_layers.append(nn.ReLU())
        
        self.encoder_fc_input_channels = (self.encoder_Conv_params['channels'][len(self.encoder_Conv_params['channels']) - 1]
                                          *(self.encoder_Conv_params['width'][len(self.encoder_Conv_params['channels']) - 1]**2))
        self.encoder_layers.append(nn.Linear(self.encoder_fc_input_channels, self.z_dim))
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
        # decoder
        self.decoder_layers = []
        self.decoder_fc_output_channels = (self.decoder_Conv_params['channels'][0]
                                          *(self.decoder_Conv_params['width'][0]**2))
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_fc_output_channels))
        for i in range(len(self.decoder_Conv_params['channels'])-1):
            if (self.decoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.decoder_Conv_params['kernel_size'][i]-1)/2)
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            elif (self.decoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels = self.decoder_Conv_params['channels'][i],
                                                          out_channels = self.decoder_Conv_params['channels'][i+1],
                                                          kernel_size = self.decoder_Conv_params['kernel_size'][i],
                                                          stride = self.decoder_Conv_params['stride'][i],
                                                          padding = padding,
                                                          output_padding = output_padding))
            self.decoder_layers.append(nn.BatchNorm2d(self.decoder_Conv_params['channels'][i+1], eps=1e-03))
            if (i < (len(self.decoder_Conv_params['channels'])-2)):
                self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        
    def forward(self, x):
        for layer_idx, layer_name in enumerate(self.encoder_layers[:-1]):
            x = layer_name(x)
        x = x.view(x.size(0), -1)
        encoded = self.encoder_layers[len(self.encoder_layers)-1](x)
        
        x = self.decoder_layers[0](encoded)
        
        x = x.view(x.size(0), self.decoder_Conv_params['channels'][0],
                         self.decoder_Conv_params['width'][0], self.decoder_Conv_params['width'][0])
        for layer_idx, layer_name in enumerate(self.decoder_layers[1:]):
            x = layer_name(x)
        reconstructed = x
        return encoded, reconstructed
    
class FMNIST_LCAE_Decoder(BaseNet):
    def __init__(self):
        super().__init__()

        self.W, self.C = 32, 1
        self.z_dim = 8
        self.decoder_Conv_params= {'channels': [256, 128, 64, self.C],
                              'kernel_size': [5, 5, 5, None],
                              'stride': [2, 2, 2, None],
                              'padding': ['SAME', 'SAME', 'SAME',  None]}
        self.decoder_Conv_params['width'] = [4, 8, 16, 32]
        
        # decoder
        self.decoder_layers = []
        self.decoder_fc_output_channels = (self.decoder_Conv_params['channels'][0]
                                          *(self.decoder_Conv_params['width'][0]**2))
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_fc_output_channels))
        for i in range(len(self.decoder_Conv_params['channels'])-1):
            if (self.decoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.decoder_Conv_params['kernel_size'][i]-1)/2)
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            elif (self.decoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels = self.decoder_Conv_params['channels'][i],
                                                          out_channels = self.decoder_Conv_params['channels'][i+1],
                                                          kernel_size = self.decoder_Conv_params['kernel_size'][i],
                                                          stride = self.decoder_Conv_params['stride'][i],
                                                          padding = padding,
                                                          output_padding = output_padding))
            self.decoder_layers.append(nn.BatchNorm2d(self.decoder_Conv_params['channels'][i+1], eps=1e-03))
            if (i < (len(self.decoder_Conv_params['channels'])-2)):
                self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        
    def forward(self, x):
        x = self.decoder_layers[0](x)
        x = x.view(x.size(0), self.decoder_Conv_params['channels'][0],
                         self.decoder_Conv_params['width'][0], self.decoder_Conv_params['width'][0])
        for layer_idx, layer_name in enumerate(self.decoder_layers[1:]):
            x = layer_name(x)
        reconstructed = x
        return reconstructed
    
class CelebA_LCAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.W, self.C = 64, 3
        self.encoder_Conv_params= {'channels': [self.C, 64, 128, 256, 512],
                                   'kernel_size': [5, 5, 5, 5, None],
                                   'stride': [2, 2, 2, 2, None],
                                   'padding': ['SAME', 'SAME', 'SAME', 'SAME', None]}
        self.encoder_Conv_params['width'] = [self.W]
        self.z_dim = 64
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
        self.encoder_layers.append(nn.Linear(self.encoder_fc_input_channels, self.z_dim))
        nn.init.kaiming_normal_(self.encoder_layers[len(self.encoder_layers)-1].weight)
        nn.init.zeros_(self.encoder_layers[len(self.encoder_layers)-1].bias)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
        # decoder
        self.decoder_layers = []
        self.decoder_fc_output_channels = (self.decoder_Conv_params['channels'][0]
                                          *(self.decoder_Conv_params['width'][0]**2))
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_fc_output_channels))
        nn.init.kaiming_normal_(self.decoder_layers[len(self.decoder_layers)-1].weight)
        nn.init.zeros_(self.decoder_layers[len(self.decoder_layers)-1].bias)
        for i in range(len(self.decoder_Conv_params)-1):
            if (self.decoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.decoder_Conv_params['kernel_size'][i]-1)/2)
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            elif (self.decoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels = self.decoder_Conv_params['channels'][i],
                                                          out_channels = self.decoder_Conv_params['channels'][i+1],
                                                          kernel_size = self.decoder_Conv_params['kernel_size'][i],
                                                          stride = self.decoder_Conv_params['stride'][i],
                                                          padding = padding,
                                                          output_padding = output_padding))
            nn.init.kaiming_normal_(self.decoder_layers[len(self.decoder_layers)-1].weight)
            nn.init.zeros_(self.decoder_layers[len(self.decoder_layers)-1].bias)
            self.decoder_layers.append(nn.BatchNorm2d(self.decoder_Conv_params['channels'][i+1], eps=1e-03))
            if (i < (len(self.decoder_Conv_params)-2)):
                self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        
    def forward(self, x):
        for layer_idx, layer_name in enumerate(self.encoder_layers[:-1]):
            x = layer_name(x)
        x = x.view(x.size(0), -1)
        encoded = self.encoder_layers[len(self.encoder_layers)-1](x)
        
        x = self.decoder_layers[0](encoded)
        
        x = x.view(x.size(0), self.decoder_Conv_params['channels'][0],
                         self.decoder_Conv_params['width'][0], self.decoder_Conv_params['width'][0])
        for layer_idx, layer_name in enumerate(self.decoder_layers[1:]):
            x = layer_name(x)
        reconstructed = x
        return encoded, reconstructed

class CelebA_LCAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.W, self.C = 64, 3
        self.encoder_Conv_params= {'channels': [self.C, 64, 128, 256, 512],
                                   'kernel_size': [5, 5, 5, 5, None],
                                   'stride': [1, 2, 2, 2, None],
                                   'padding': ['SAME', 'SAME', 'SAME', 'SAME', None]}
        self.encoder_Conv_params['width'] = [self.W]
        self.z_dim = 64
        self.decoder_Conv_params= {'channels': [512, 256, 128, 64, self.C],
                              'kernel_size': [5, 5, 5, 5, None],
                              'stride': [2, 2, 2, 1, None],
                              'padding': ['SAME', 'SAME', 'SAME', 'SAME', None]}
        self.decoder_Conv_params['width'] = [8, 16, 32, 64, 64]
        
        # decoder
        self.decoder_layers = []
        self.decoder_fc_output_channels = (self.decoder_Conv_params['channels'][0]
                                          *(self.decoder_Conv_params['width'][0]**2))
        self.decoder_layers.append(nn.Linear(self.z_dim, self.decoder_fc_output_channels))
        nn.init.kaiming_normal_(self.decoder_layers[len(self.decoder_layers)-1].weight)
        nn.init.zeros_(self.decoder_layers[len(self.decoder_layers)-1].bias)
        for i in range(len(self.decoder_Conv_params)-1):
            if (self.decoder_Conv_params['padding'][i]=='SAME'):
                padding = int((self.decoder_Conv_params['kernel_size'][i]-1)/2)
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            elif (self.decoder_Conv_params['padding'][i]=='VALID'):
                padding = 0
                output_padding = (self.decoder_Conv_params['width'][i+1]
                                  -self.decoder_Conv_params['stride'][i]*(self.decoder_Conv_params['width'][i]-1)
                                  -self.decoder_Conv_params['kernel_size'][i]
                                  +2*padding)
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels = self.decoder_Conv_params['channels'][i],
                                                          out_channels = self.decoder_Conv_params['channels'][i+1],
                                                          kernel_size = self.decoder_Conv_params['kernel_size'][i],
                                                          stride = self.decoder_Conv_params['stride'][i],
                                                          padding = padding,
                                                          output_padding = output_padding))
            nn.init.kaiming_normal_(self.decoder_layers[len(self.decoder_layers)-1].weight)
            nn.init.zeros_(self.decoder_layers[len(self.decoder_layers)-1].bias)
            self.decoder_layers.append(nn.BatchNorm2d(self.decoder_Conv_params['channels'][i+1], eps=1e-03))
            if (i < (len(self.decoder_Conv_params)-2)):
                self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        
    def forward(self, x):
        x = self.decoder_layers[0](x)
        x = x.view(x.size(0), self.decoder_Conv_params['channels'][0],
                         self.decoder_Conv_params['width'][0], self.decoder_Conv_params['width'][0])
        for layer_idx, layer_name in enumerate(self.decoder_layers[1:]):
            x = layer_name(x)
        reconstructed = x
        return reconstructed
