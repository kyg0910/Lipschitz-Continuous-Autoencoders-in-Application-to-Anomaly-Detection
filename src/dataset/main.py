def load_dataset(dataset_name, data_path, normal_class, contam_ratio,random_seed):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fmnist', 'kdd99')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist' :
        from .mnist import MNIST_Dataset
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class,contam_ratio=contam_ratio,random_seed=random_seed)

    if dataset_name =='fmnist':
        from .fmnist import FMNIST_Dataset
        dataset = FMNIST_Dataset(root=data_path, normal_class=normal_class,contam_ratio=contam_ratio,random_seed=random_seed)
        
    if dataset_name =='kdd99':
        from .kdd99 import KDD_Dataset
        dataset = KDD_Dataset(root=data_path, contam_ratio=contam_ratio,random_seed=random_seed)
        
    return dataset


