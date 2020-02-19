import importlib

lcae = dict(
    kdd99  = dict(
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    sigma=1.0,
    scale_list= [.2, .5, 1., 2., 5.],
    lipschitz_constant=0.8,
    mmd_weight=2.0,
    lipschitz_penalty_weight=2.0  ,
    dataset_name= 'KDD',
    net_name= 'KDD_LCAE',
    data_path= '../../data_kdd99',
    n_epochs= 50,
    batch_size=50,
    train_ratio= 50,
    test_ratio= 25,
    val_ratio=25,
    mmd_batch_size=50,
    z_dim=5,    
    )      ,

    fmnist = dict(
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    sigma=1.0,
    scale_list= [.2, .5, 1., 2., 5.],
    lipschitz_constant=0.8,
    mmd_weight=2.0,
    lipschitz_penalty_weight=2.0  ,
    dataset_name= 'fmnist',
    net_name= 'FMNIST_LCAE',
    data_path= '../../data_fmnist',
    n_epochs= 100,
    batch_size=100,
    train_ratio= 60,
    test_ratio= 20,
    val_ratio=20,
    mmd_batch_size=100,
    z_dim=8,    
    )      ,

    mnist = dict(
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    sigma=1.0,
    scale_list= [.2, .5, 1., 2., 5.],
    lipschitz_constant=0.8,
    mmd_weight=2.0,
    lipschitz_penalty_weight=2.0  ,
    dataset_name= 'mnist',
    net_name= 'MNIST_LCAE',
    data_path= '../../data_mnist',
    n_epochs= 100,
    batch_size=100,
    train_ratio= 60,
    test_ratio= 20,
    val_ratio=20,
    mmd_batch_size=100,
    z_dim=8,    
    )   ,   


    celeba= dict(
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    sigma=1.0,
    scale_list= [.2, .5, 1., 2., 5.],
    lipschitz_constant=0.8,
    mmd_weight=2.0,
    lipschitz_penalty_weight=2.0 , 
    dataset_name= 'celeba',
    net_name= 'CelebA_LCAE',
    data_path= '../../data_celeba',
    csv_file= '../data_celeba/list_attr_celeba.csv', 
    root_dir='../data_celeba/img_align_celeba/', 
    n_epochs= 100,
    batch_size=100,
    train_ratio= 50,
    test_ratio= 25,
    val_ratio=25,
    mmd_batch_size=100,
    z_dim=64,    
    )
)

svdd = dict(
    kdd99 = dict(
    load_config = None,
    load_model = None,
    objective = 'one-class',
    nu = 0.1,
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    dataset_name= 'KDD',
    net_name= 'KDD_LeNet',
    data_path= '../data_kdd',
    n_epochs= 50,
    batch_size=50,
    train_ratio= 50,
    test_ratio= 25,
    val_ratio=25,
    z_dim=15           
    ),


    fmnist = dict(
    load_config = None,
    load_model = None,
    objective = 'one-class',
    nu = 0.1,
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    dataset_name= 'fmnist',
    net_name= 'fmnist_LeNet',
    data_path= '../data_fmnist',
    n_epochs= 50,
    batch_size=100,
    train_ratio= 60,
    test_ratio= 20,
    val_ratio=20,
    z_dim=32           
    ),

    mnist = dict(
    load_config = None,
    load_model = None,
    objective = 'one-class',
    nu = 0.1,
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    dataset_name= 'mnist',
    net_name= 'mnist_LeNet',
    data_path= '../data_mnist',
    n_epochs= 50,
    batch_size=100,
    train_ratio= 60,
    test_ratio= 20,
    val_ratio=20,
    z_dim=32           
    ),

    celeba = dict(
    load_config = None,
    load_model = None,
    objective = 'one-class',
    nu = 0.1,
    optimizer_name= 'adam' ,
    lr =2e-4,
    n_jobs_dataloader=0,
    data_path='../../data_celeba',
    csv_file='../data_celeba/list_attr_celeba.csv',
    root_dir='../data_celeba/img_align_celeba/',
    dataset_name= 'celeba',
    net_name= 'CelebA_LeNet',
    n_epochs= 50,
    batch_size=100,
    train_ratio= 50,
    test_ratio= 25,
    val_ratio=25,
    z_dim=64           
    )
)

alad = dict(
    
    kdd99 = dict(
    train_ratio= 50,
    test_ratio= 25,
    val_ratio=25,        
    ),


    fmnist = dict(
    train_ratio= 60,
    test_ratio= 20,
    val_ratio=20     
    ),

    mnist = dict(
    train_ratio= 60,
    test_ratio= 20,
    val_ratio=20
    ),

    celeba = dict(
    train_ratio= 50,
    test_ratio= 25,
    val_ratio=25
    )
)

def load_config(model, dataset):

    if model == 'lcae':
        config = lcae[dataset]
        lcae[dataset].update(c=2.0 * lcae[dataset]['z_dim'] * (lcae[dataset]['sigma']**2))
    elif model == 'svdd':
        config = svdd[dataset]  
    elif model == 'alad':
        config = alad[dataset] 
    return config
