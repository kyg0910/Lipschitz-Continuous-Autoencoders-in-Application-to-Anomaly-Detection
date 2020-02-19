import errno, logging, os, sys, torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils import get_target_label_idx, global_contrast_normalization
from base import BaseADDataset

class KDD_Dataset(BaseADDataset):
    def __init__(self, root: str, contam_ratio, random_seed):
        super().__init__(root)
        self.contam_ratio = contam_ratio
        self.random_seed = random_seed
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = tuple([1])
        _, _, test_x, test_y, _, _, train_actual_x, val_actual_x = rescale_KDD99(50, 25, 25, self.contam_ratio, self.random_seed)
        y = np.zeros(np.shape(train_actual_x)[0])
        self.testset = np.column_stack([test_x,test_y])
        self.trainset = np.column_stack([train_actual_x,y])
        y = np.zeros(np.shape(val_actual_x)[0])
        self.valset = np.column_stack([val_actual_x,y])
        
        self.train_set = KDD99(data=self.trainset)
        # Subset train_set to normal class
        self.test_set = KDD99(data=self.testset)
        self.val_set = KDD99(data=self.valset)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_test)
        return train_loader, test_loader, val_loader
    
class KDD99(Dataset):
    def __init__(self, data):
        self.data_info = data
        # First column contains the image paths
        self.attr_arr = self.data_info[:,:-1].astype(float)
        # Second column is the labels
        self.label_arr = self.data_info[:,-1].astype(float)
        # Calculate len
        self.data_len = len(self.label_arr)
        
    def __len__(self):
        return len(self.label_arr)
    
    def __getitem__(self, idx):
        single_attr = self.attr_arr[idx]
        single_label = self.label_arr[idx]
        
        return single_attr, single_label, idx

root = '../data_kdd99'
folder = 'kddcup.data_10_percent'
data_file = 'kddcup.data_10_percent'
urls = ['http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz']

def _check_exists_KDD():
    return os.path.exists(os.path.join(root, folder, data_file))

def download_KDD99():
    """Download the KDD99 data in current directory if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import gzip
    if _check_exists_KDD():
        print("The file already exists!")
        return
    # download files
    try:
        os.makedirs(os.path.join(root, folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    for url in urls:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)
    print('Done!')
    
# Preprocessing the data
file = 'preprocess.npy'

def _check_exists_preprocess():
    return os.path.exists(os.path.join(root, file))

def preprocess_KDD99():
    if _check_exists_preprocess():
        print("The file already exists!")
        return
    else:
        col_names = np.array(["duration", "protocol_type", "service", "flag", "src_bytes",
                              "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                              "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                              "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                              "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                              "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                              "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                              "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                              "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                              "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels"])
        nominal_inx = [1, 2, 3]
        binary_inx = [6, 11, 13, 14, 20, 21]
        numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))
        nominal_cols = col_names[nominal_inx].tolist()
        binary_cols = col_names[binary_inx].tolist()
        numeric_cols = col_names[numeric_inx].tolist()
        df = pd.read_csv(os.path.join(root, folder, data_file), header=None, names=col_names)
        df = np.array(df)
        nomial = [1, 2, 3, 6, 11, 13, 14, 20, 21]
        label = np.zeros((np.shape(df)[0], 1))
        label[df[:, 41] != 'normal.', 0] = 1.0
        for i in range(len(nomial)):
            values = np.array(df[:, nomial[i]])
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(values)
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            onehot_encoded = onehot_encoded[:, :-1]
            df = np.hstack((df, onehot_encoded))
        df = np.hstack((df, label))
        df = np.delete(df, 41, axis=1)
        df = np.delete(df, nomial, axis=1)
        df[:, -1] = 1 - df[:, -1]
        np.save('../data_kdd99/preprocess.npy', df)
        print("saved 'preprocess.npy'")


def printout_parameters_preprocess(train_ratio, test_ratio, val_ratio, contam_ratio, random_seed):
    logging.info("""
    ######## splitting parameters ########
    TRAIN_RATIO = {}
    TEST_RATIO = {}
    VAL_RATIO = {}
    CONTAM_RATIO = {}
    """.format(train_ratio, test_ratio, val_ratio, contam_ratio))

def rescale_KDD99(train_ratio, test_ratio, val_ratio, contam_ratio, random_seed):
    printout_parameters_preprocess(train_ratio, test_ratio, val_ratio, contam_ratio, random_seed)
    train_idx, train_contam_idx, test_idx, val_idx, val_contam_idx, max_scaler, min_scaler = None, None, None, None, None, None, None
    np.random.seed(random_seed)

    print("Checking if the data is available...")
    download_KDD99()
    preprocess_KDD99()

    try:
        data = np.load('../data_kdd99/preprocess.npy')
    except FileNotFoundError as e:
        print(e, "Make sure you download KDD99 and preprocess it!")
    else:
        try:
            if train_ratio <= 0 or val_ratio < 0 or test_ratio <= 0 or contam_ratio < 0 or isinstance(contam_ratio, int) != True or contam_ratio > 100:
                raise Exception
        except Exception:
            print("[Error] put non-negative values!(train and test ratio should not be zero)")
            sys.exit(1)

    ratio_tuple = np.array([train_ratio, val_ratio, test_ratio])
    train_ratio_dec, val_ratio_dec, test_ratio_dec = np.true_divide(ratio_tuple, np.sum(ratio_tuple))

    n_data = len(data)
    train_idx = np.random.choice(n_data, int(train_ratio_dec * n_data), replace=False)
    test_idx = np.setdiff1d(np.arange(n_data), train_idx)
    train = data[train_idx, :]

    n_train_normal = np.sum(train[:, -1] == 0)

    val_idx = np.random.choice(test_idx, int(val_ratio_dec * n_data), replace=False)
    test_idx = np.setdiff1d(test_idx, val_idx)

    train_array = np.arange(train.shape[0])
    train_contam_idx = train_array[train[:, -1] == 0]
    if contam_ratio != 0:
        contam_ratio_dec = contam_ratio / 100
        train_contam = np.random.choice(train_array[train[:, -1] == 1],
                                        int(n_train_normal * contam_ratio_dec / (1 - contam_ratio_dec)),
                                        replace=False)
        train_contam_idx = np.append(train_array[train[:, -1] == 0], train_contam)

    train_contam = train[train_contam_idx, :]
    max_scaler = np.max(train_contam, axis=0)[:-1]
    min_scaler = np.min(train_contam, axis=0)[:-1]

    train_x, train_y, test_x, test_y, val_x, val_y, train_actual_x, val_actual_x = None, None, None, None, None, None, None, None

    degenerated_index = (max_scaler == min_scaler)

    train_x, test_x = data[train_idx, :], data[test_idx, :]
    train_x, train_y, test_x, test_y = train_x[:, :-1], train_x[:, -1], test_x[:, :-1], test_x[:, -1]

    train_x[:, degenerated_index] = train_x[:, degenerated_index] - min_scaler[degenerated_index]
    train_x[:, ~degenerated_index] = 2.0 * (train_x[:, ~degenerated_index] - min_scaler[~degenerated_index]) / (max_scaler[~degenerated_index] - min_scaler[~degenerated_index]) - 1.0

    test_x[:, degenerated_index] = test_x[:, degenerated_index] - min_scaler[degenerated_index]
    test_x[:, ~degenerated_index] = 2.0 * (test_x[:, ~degenerated_index] - min_scaler[~degenerated_index]) / (max_scaler[~degenerated_index] - min_scaler[~degenerated_index]) - 1.0
    train_actual_x = train_x[train_y == 0, :]
    logging.info('-'*50)
    logging.info('train anomaly ratio :{}'.format(sum(train_y)/len(train_y)))
    logging.info('test anomaly ratio :{}'.format(sum(test_y)/len(test_y)))
    if val_ratio != 0:
        val = data[val_idx, :]
        n_val_normal = np.sum(val[:, -1] == 0)

        val_array = np.arange(val.shape[0])
        val_contam_idx = val_array[val[:, -1] == 0]

        val_x = data[val_idx, :]
        val_y = val_x[:, -1]
        val_x = val_x[:, :-1]
        val_x[:, degenerated_index] = val_x[:, degenerated_index] - min_scaler[degenerated_index]
        val_x[:, ~degenerated_index] = 2.0 * (val_x[:, ~degenerated_index] - min_scaler[~degenerated_index]) / (max_scaler[~degenerated_index] - min_scaler[~degenerated_index]) - 1.0
        val_actual_x = val_x[val_y == 0, :]
        logging.info('validation anomaly ratio :{}'.format(sum(val_y)/len(val_y)))
    logging.info('-'*50)

    if contam_ratio != 0:
        val_contam = np.random.choice(val_array[val[:, -1] == 1],
                                      int(n_val_normal * contam_ratio_dec / (1 - contam_ratio_dec)),
                                      replace=False)
        val_contam_idx = np.append(val_array[val[:, -1] == 0], val_contam)
        train_actual_x = train_x[train_contam_idx, :]
        val_actual_x = val_x[val_contam_idx, :]

    return train_x, train_y, test_x, test_y, val_x, val_y, train_actual_x, val_actual_x

def get_shape_input():
    return (None, 115)
