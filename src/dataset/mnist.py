import codecs, errno, math, os, pickle, shutil, tarfile, torch, urllib3
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset,Subset 
from torchvision.datasets import MNIST
from base import BaseADDataset

class MNIST_Dataset(BaseADDataset):

    def __init__(self, root: str, normal_class=3, contam_ratio=0,random_seed=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        transform = transforms.Compose([transforms.ToTensor()])
        centered,normalize = False, True        
        trainx, trainy = get_train(normal_class,centered,normalize,contam_ratio,random_seed)
        validx, validy =get_valid(normal_class,centered,normalize,contam_ratio,random_seed)
        testx, testy = get_test(normal_class,centered,normalize,contam_ratio,random_seed)
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        
        resized_trainx = np.zeros((np.shape(trainx)[0], 32, 32, 1))
        for i in range(np.shape(trainx)[0]):
            resized_trainx[i, :, :, 0] = resize(trainx[i, :, :], (32, 32))
        resized_validx = np.zeros((np.shape(validx)[0], 32, 32, 1))
        for i in range(np.shape(validx)[0]):
            resized_validx[i, :, :, 0] = resize(validx[i, :, :], (32, 32))
        resized_testx = np.zeros((np.shape(testx)[0], 32, 32, 1))
        for i in range(np.shape(testx)[0]):
            resized_testx[i, :, :, 0] = resize(testx[i, :, :], (32, 32))
        
        trainx, validx, testx = resized_trainx, resized_validx, resized_testx
        
        self.train_set = MyMNIST(trainx, trainy, transform=transform, target_transform=target_transform)
        self.val_set = MyMNIST(validx, validy, transform=transform, target_transform=target_transform)
        self.test_set = MyMNIST(testx, testy, transform=transform, target_transform=target_transform)
        
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_test)
        return train_loader, test_loader, val_loader

class MyMNIST(Dataset):
    def __init__(self, x,y,transform, target_transform):
        # First column contains the image paths
        self.x = x
        # Second column is the labels
        self.y = y
        # Calculate len
        self.data_len = len(self.y)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        img, target = self.x[index] , self.y[index]               
        if self.transform is not None:
            a,b = img.shape[0], img.shape[1]
            img = img.reshape(a,b,1)
            img = self.transform(img)
        return img, target, index

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

def get_train(label=-1, centered=True, normalize=True,contam_ratio=0,random_seed=0):
    return _get_adapted_dataset("train", label, centered, normalize,contam_ratio,random_seed)

def get_test(label=-1, centered=True, normalize=True,contam_ratio=0,random_seed=0):
    return _get_adapted_dataset("test", label, centered, normalize,contam_ratio,random_seed)

def get_valid(label=-1, centered=True, normalize=True,contam_ratio=0,random_seed=0):
    return _get_adapted_dataset("valid", label, centered, normalize,contam_ratio,random_seed)
    
def get_shape_input():
    return (None, 28, 28)

def get_shape_input_flatten():
    return (None, 28*28)

def get_shape_label():
    return (None,)

def num_classes():
    return 10

def get_anomalous_proportion():
    return 0.9

def _check_exists():
    root = "../data_mnist"
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    

    return os.path.exists(os.path.join(root, processed_folder, training_file)) and \
        os.path.exists(os.path.join(root, processed_folder, test_file))

def download():
    root = "../data_mnist"
    raw_folder = 'raw'    
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    """Download the MNIST data if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import gzip

    if _check_exists():
        return

    # download files
    try:
        os.makedirs(os.path.join(root, raw_folder))
        os.makedirs(os.path.join(root, processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for url in urls:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)

    # process and save as torch files
    print('Processing...')
    training_set = (
        read_image_file(os.path.join(root, raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(root, raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join(root, raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(root, raw_folder, 't10k-labels-idx1-ubyte'))
    )
    with open(os.path.join(root, processed_folder, training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(root, processed_folder, test_file), 'wb') as f:
        torch.save(test_set, f)

    print('Done!')

def _get_dataset(centered=False, normalize=False):
    '''
    Gets the adapted dataset for the experiments
    Args : 
            split (str): train or test
            normalize (bool): (Default=True) normalize data
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns : 
            (tuple): <training, testing> images and labels
    '''
    
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    root = "../data_mnist"    
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    
    if not _check_exists():
        download()
        
    train_data, train_labels = torch.load(os.path.join(root, processed_folder, training_file))
    test_data, test_labels = torch.load(os.path.join(root, processed_folder, test_file))

    imgs = np.concatenate((train_data,test_data),axis = 0)
    lbls = np.concatenate((train_labels ,test_labels),axis = 0)
    
    # Convert images to [0..1] range
    if normalize:
        imgs = imgs.astype(np.float32)/255.0
    if centered:
        imgs = imgs.astype(np.float32)*2. - 1.
    return imgs.astype(np.float32), lbls

def _get_adapted_dataset(split, label=None, centered=False, normalize=False, contam_ratio=0,random_seed=0):
    """
    Gets the adapted dataset for the experiments
    Args : 
            split (str): train or test
            mode (str): inlier or outlier
            label (int): int in range 0 to 10, is the class/digit
                         which is considered inlier or outlier
            rho (float): proportion of anomalous classes INLIER
                         MODE ONLY
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns : 
            (tuple): <training, testing> images and labels
    """
    dataset = {}
    full_x_data, full_y_data = _get_dataset(centered=centered, normalize=normalize)
    full_y_data[full_y_data == 10] = 0
    
    dataset['x_train'], dataset['x_test'], \
    dataset['y_train'], dataset['y_test'] = train_test_split(full_x_data,
                                                             full_y_data,
                                                             test_size=0.2,
                                                             random_state=random_seed)
    
    dataset['x_train'], dataset['x_valid'], \
    dataset['y_train'], dataset['y_valid'] = train_test_split(dataset['x_train'],
                                                             dataset['y_train'],
                                                             test_size=0.25,
                                                             random_state=random_seed)
    
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if label != -1:
        if split in ['train', 'valid']:
            inliers = dataset[key_img][dataset[key_lbl] == label], \
                      dataset[key_lbl][dataset[key_lbl] == label]
            outliers = dataset[key_img][dataset[key_lbl] != label], \
                       dataset[key_lbl][dataset[key_lbl] != label]

            if contam_ratio != 0:
                contam_ratio_dec = contam_ratio/100
                n_data = dataset[key_img][dataset[key_lbl] == label].shape[0]                
                contam_idx = np.random.choice(n_data, int(n_data * contam_ratio_dec / (1 - contam_ratio_dec)),
                                            replace=False)                
                dataset[key_img] =np.concatenate((dataset[key_img][dataset[key_lbl] == label], dataset[key_img][dataset[key_lbl] != label][contam_idx]),axis = 0)
                dataset[key_lbl] =np.concatenate((dataset[key_lbl][dataset[key_lbl] == label], dataset[key_lbl][dataset[key_lbl] != label][contam_idx]),axis = 0)
                dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl],label)
            else:
                dataset[key_img], dataset[key_lbl] = inliers

            return (dataset[key_img], dataset[key_lbl])
        else:
            dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl],
                                                         label)
            return (dataset[key_img], dataset[key_lbl])
        
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
