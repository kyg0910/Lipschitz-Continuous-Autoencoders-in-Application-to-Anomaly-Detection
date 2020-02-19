from __future__ import print_function, division
import errno, logging, math, os, pickle, shutil, tarfile, torch, urllib3
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from base import BaseADDataset
from utils import *

class CelebA_Dataset(BaseADDataset):
    def __init__(self, csv_file, root_dir , attribute, normal_class, train_ratio, test_ratio, val_ratio, contam_ratio, random_seed, transform=None):
        super().__init__(root_dir)
        
        np.random.seed(random_seed)
        
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = tuple([1])

        crop = CenterCrop(140)
        scale = Rescale(64)
        transform = transforms.Compose([crop, scale, transforms.ToTensor()])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        whole_set = MyCelebA(csv_file, root_dir, attribute, normal_class, transform)

        # Subset train_set to normal class
        train_ratio_dec = train_ratio/(train_ratio+val_ratio+test_ratio)
        test_ratio_dec = test_ratio/(train_ratio+val_ratio+test_ratio)
        val_ratio_dec = val_ratio/(train_ratio+val_ratio+test_ratio)
        RNG = np.random.RandomState(random_seed)
        
        idx = np.load('../data_celeba/celeba_index_male_25000.npy')
        idx = RNG.permutation(idx)
        logging.info('whole index : {} , length : {}'.format(idx,len(idx)))

        y = whole_set.label_arr        
        
        train_idx = idx[:int(len(idx)*(1-test_ratio_dec))]
        test_idx = idx[int(len(idx)*(1-test_ratio_dec)):]
        # train : val = 50% : 25%
        val_idx = train_idx[int(len(train_idx)*(train_ratio/(train_ratio+val_ratio))):]
        train_idx = train_idx[:int(len(train_idx)*(train_ratio/(train_ratio+val_ratio)))]

        ### remove anomalies on train/val
        train_normal_idx = train_idx[y[train_idx] == 0]
        val_normal_idx = val_idx[y[val_idx] == 0]
        logging.info('whole train index : {} , length : {}'.format(train_normal_idx, len(train_normal_idx)))

        if contam_ratio != 0: 
            n_data = len(train_normal_idx)
            contam_ratio_dec = contam_ratio / 100
            train_contam = np.random.choice(train_idx[y[train_idx]!=0],
                                            int(n_data * contam_ratio_dec / (1 - contam_ratio_dec)),
                                            replace=False)
            train_normal_idx = np.append(train_normal_idx, train_contam)
            n_data = len(val_normal_idx)
            contam_ratio_dec = contam_ratio / 100
            val_contam = np.random.choice(val_idx[y[val_idx] != 0],
                                            int(n_data * contam_ratio_dec / (1 - contam_ratio_dec)),
                                            replace=False)
            val_normal_idx = np.append(val_normal_idx, val_contam)
        
        self.train_normal_idx = train_normal_idx
        self.test_idx = test_idx
        self.val_normal_idx =  val_normal_idx
        
        self.train_set = Subset(whole_set, train_normal_idx)
        self.test_set = Subset(whole_set, test_idx)
        self.val_set = Subset(whole_set, val_normal_idx)
        
        
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_test)
        return train_loader, test_loader, val_loader

class MyCelebA(Dataset):
    def __init__(self, csv_file, root_dir, attribute, normal_class, transform =None):
        self.all_attributes_frame = pd.read_csv(csv_file).values
        self.root_dir = root_dir
        self.transform = transform
        # First column contains the image paths
        # Second column is the labels
        
        self.label_arr = self.all_attributes_frame[:,attribute+1]
        self.label_arr_male = self.all_attributes_frame[:,21]
        
        if normal_class == -1:
            self.label_arr[self.label_arr == -1] = 0
        else:
            self.label_arr[self.label_arr == 1] = 0
            self.label_arr[self.label_arr == -1] = 1
        self.transform = transform
        
        # Calculate len
        self.data_len = len(self.label_arr)
        
    def __len__(self):
        return len(self.all_attributes_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.all_attributes_frame[idx, 0])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)
        label = self.label_arr[idx]
        
        return image , label, idx

def adapt_labels_outlier_task(true_labels, label):
    """Adapt labels to anomaly detection context

    Args :
            true_labels (list): list of ints
            label (int): label which is considered inlier
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    if label == 1:
        (true_labels[true_labels == label], true_labels[true_labels != label]) = (1, 0)
        true_labels = [1] * true_labels.shape[0] - true_labels
    else:
        (true_labels[true_labels != label], true_labels[true_labels == label]) = (1, 0)
    return true_labels

def get_dataset(attributes = 20, normal_class=1, centered=True, normalize=True,contam_ratio=0,random_seed=0):
    path = '../data_celeba/CelebA_64resol.npy'
    x = np.load(path)
    path = '../data_celeba/list_attr_celeba.csv'
    df_attributes = pd.read_csv(path).values[:,1:]
    y = df_attributes[:, attributes]
    if normal_class == -1:
        y[y==-1] = 0
    else:
        y[y==1] = 0
        y[y==-1] = 1
    train_ratio = 50
    test_ratio = 25
    val_ratio = 25
    
    np.random.seed(random_seed)

    RNG = np.random.RandomState(random_seed)    
    
    idx = np.load('../data_celeba/celeba_index_male_25000.npy')
    idx = RNG.permutation(idx)
    logging.info('whole index : {} , length : {}'.format(idx,len(idx)))
    #gender conditioning
    
    train_ratio_dec = train_ratio/(train_ratio+val_ratio+test_ratio)
    test_ratio_dec = test_ratio/(train_ratio+val_ratio+test_ratio)
    val_ratio_dec = val_ratio/(train_ratio+val_ratio+test_ratio)

    train_idx = idx[:int(len(idx)*(1-test_ratio_dec))]
    test_idx = idx[int(len(idx)*(1-test_ratio_dec)):]

    # train : val = 50% : 25%
    val_idx = train_idx[int(len(train_idx)*(train_ratio/(train_ratio+val_ratio))):]
    train_idx = train_idx[:int(len(train_idx)*(train_ratio/(train_ratio+val_ratio)))]

    ### remove anomalies on train/val
    train_normal_idx = train_idx[y[train_idx] == 0]
    val_normal_idx = val_idx[y[val_idx] == 0]
    logging.info('whole train index : {} , length : {}'.format(train_normal_idx, len(train_normal_idx)))

    if contam_ratio != 0: 
        n_data = len(train_normal_idx)
        contam_ratio_dec = contam_ratio / 100
        train_contam = np.random.choice(train_idx[y[train_idx]!=0],
                                        int(n_data * contam_ratio_dec / (1 - contam_ratio_dec)),
                                        replace=False)
        train_normal_idx = np.append(train_normal_idx, train_contam)
        n_data = len(val_normal_idx)
        contam_ratio_dec = contam_ratio / 100
        val_contam = np.random.choice(val_idx[y[val_idx] != 0],
                                        int(n_data * contam_ratio_dec / (1 - contam_ratio_dec)),
                                        replace=False)
        val_normal_idx = np.append(val_normal_idx, val_contam)
    trainx,testx,valx = x[train_normal_idx],x[test_idx],x[val_normal_idx]
    trainy,testy,valy = y[train_normal_idx],y[test_idx],y[val_normal_idx]
    del x
    return trainx,trainy,testx,testy,valx,valy
    
def get_shape_input():
    return (None, 64, 64, 3)    
