import argparse, importlib, json, logging, os, random, time, torchvision
import numpy as np
import torchvision.transforms as transforms
from dataset.main import load_dataset
from dataset.celeba import CelebA_Dataset
from base import BaseADDataset
from architecture import build_network, build_decoder_network
from LCAE_trainer import LCAETrainer
from utils import *
import configuration

class LCAE(object):
    def __init__(self, dataset_name: str , sigma: float, mmd_batch_size: int, z_dim: int, scale_list: tuple , c: float 
                 ,lipschitz_constant: float , mmd_weight: float, lipschitz_penalty_weight: float,contam_ratio: int,
                 random_seed: int, normal_class: int, attribute: int, saving_directory: str):

        self.dataset_name = dataset_name
        self.sigma = sigma
        self.mmd_batch_size = mmd_batch_size
        self.z_dim = z_dim
        self.scale_list =scale_list
        self.c = c
        self.lipschitz_constant = lipschitz_constant
        self.mmd_weight = mmd_weight
        self.lipschitz_penalty_weight =lipschitz_penalty_weight
        self.contam_ratio = contam_ratio
        self.random_seed = random_seed
        self.normal_class = normal_class
        self.attribute = attribute
        self.saving_directory = saving_directory
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.net_name = None
        self.net = None  # neural network \phi

        self.net_decoder_name = None
        self.net_decoder = None 
        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)
        print(build_network(net_name))
        
    def set_decoder_network(self, net_name):
        """Builds the neural network \phi."""
        if net_name != 'KDD_LCAE':
            self.net_decoder_name = net_name
            self.net_decoder = build_decoder_network(net_name)
            print(build_decoder_network(net_name))
        else:
            self.net_decoder_name = net_name
            self.net_decoder = build_network(net_name)
            print(build_network(net_name))
   
    def train(self, dataset: BaseADDataset, optimizer_name: str, lr: float, n_epochs: int, batch_size: int , device: str):
        self.optimizer_name = optimizer_name
        self.lr = lr
        
        self.trainer = LCAETrainer(self.dataset_name, self.optimizer_name, self.lr, n_epochs, batch_size,
                                   device, self.sigma, self.mmd_batch_size, self.z_dim, self.scale_list, self.c, 
                                   self.lipschitz_constant, self.mmd_weight, self.lipschitz_penalty_weight, self.contam_ratio,
                                   self.random_seed, self.normal_class, self.attribute, self.saving_directory)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str, show_tsne: bool):
        if self.trainer is None:
            self.trainer = LCAETrainer(device=device)
        
        L1 = self.trainer.test(dataset, self.net, self.net_decoder, show_tsne)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        
        return L1

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str, lr: float, n_epochs: int ,
                 batch_size: int, device: str ):
        self.ae_optimizer_name = optimizer_name
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()

        # Filter out decoder network keys
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

def run(args):
    ################################################################################
    # Settings: logging and declare variables
    ################################################################################
    # for variables
    CONFIG_DICT = configuration.load_config(args.model,args.dataset)
    globals().update(CONFIG_DICT)
 
    directory = make_dir_to_save_results(args.model,args.dataset,args.contamratio,args.mmdweight, 
                                         args.lipschitzweight,args.lipschitzconstant)
    if args.dataset == 'kdd99':
        outcome_text_file = '{}/outcome_{}_{}_mmd weight={}_lipschitz weight={}_lipschitz constant={}_contam ratio={}_replications={}.txt'.format(directory,args.model,args.dataset,args.mmdweight, args.lipschitzweight,args.lipschitzconstant,args.contamratio,args.number)
        log_name = 'LCAE' + '_' + '{}_mmd weight={}_lipschitz weight={}_lipschitz constant={}_contam ratio={}_'.format(args.dataset,  args.mmdweight, args.lipschitzweight, args.lipschitzconstant,args.contamratio) + 'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    elif (args.dataset == 'mnist') or (args.dataset == 'fmnist') :
        outcome_text_file = '{}/outcome_{}_{}_mmd weight={}_lipschitz weight={}_lipschitz constant={}_contam ratio={}_normal class={}_replications={}.txt'.format(directory, args.model, args.dataset,
                              args.mmdweight, args.lipschitzweight,
                              args.lipschitzconstant, args.contamratio, args.normalclass, args.number)
        log_name = 'LCAE' + '_' + '{}_mmd weight={}_lipschitz weight={}_lipschitz constant={}_contam ratio={}_normal class={}_'.format(args.dataset,  args.mmdweight, args.lipschitzweight, args.lipschitzconstant,args.contamratio,args.normalclass) + 'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    elif args.dataset == 'celeba' :
        outcome_text_file = '{}/outcome_{}_{}_mmd weight={}_lipschitz weight={}_lipschitz constant={}_contam ratio={}_attribute={}_normal class={}_replications={}.txt'.format(directory, args.model, args.dataset,args.mmdweight,
                                              args.lipschitzweight, args.lipschitzconstant,
                                              args.contamratio, args.attribute, args.normalclass, args.number)
        log_name = 'LCAE' + '_' + '{}_mmd weight={}_lipschitz weight={}_lipschitz constant={}_contam ratio={}_normal class={}_attribute={}_'.format(args.dataset,  args.mmdweight, args.lipschitzweight, args.lipschitzconstant,args.contamratio,args.normalclass,args.attribute) + 'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
        
    # For logging(to be fixed)
    if os.path.exists('../loggings/lcae/{}'.format(args.dataset)) is False:
        os.makedirs('../loggings/lcae/{}'.format(args.dataset))

    

    if os.path.exists('../loggings/lcae/{}/{}.log'.format(args.dataset, log_name)) is True:
        os.remove('../loggings/lcae/{}/{}.log'.format(args.dataset, log_name))
    logging.basicConfig(filename='../loggings/lcae/{}/{}.log'.format(args.dataset,log_name),
                        level=logging.INFO)
    stderrLogger = logging.StreamHandler()
    stderrLogger.setFormatter(
        logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'))
    logging.getLogger().addHandler(stderrLogger)
    logging.info('File saved: {}'.format(log_name))
    
    with open('{}'.format(outcome_text_file), 'w') as f:
        f.write('Corresponding log_name : {}'.format(log_name))
        f.write('7 metrics are ' + '[accuracy, specificity, precision, recall, f1, roc_auc, auprc]'+'\n')
    
    store_performance = []
    if args.decide == 'int':
        args.number = int(args.number)
    else:
        args.number = list(map(int, args.number.split(',')))

    if type(args.number) is int:
        seed_tuple = range(args.number)
    elif type(args.number) is list:
        seed_tuple = args.number
        
    for random_seed in seed_tuple:
        ################################################################################
        # TRAIN
        ################################################################################
        # Load the data
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        device = 'cuda'
        if not torch.cuda.is_available():
            device = 'cpu'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        torch.cuda.set_device(args.gpu)
        logging.info('GPU NUM: %d' % args.gpu)
        logging.info('###########Data Property#############')
        logging.info('Train ratio: %d' % train_ratio)
        logging.info('Test ratio: %d' % test_ratio)
        logging.info('Validation ratio: %d' % val_ratio)
        logging.info('Contamination ratio: %d' % args.contamratio)
        logging.info('Random seed: %d' % random_seed)
        if args.dataset != 'kdd99':
            logging.info('Normal class: %d' % args.normalclass)
            if args.dataset == 'celeba':
                logging.info('Attribute for CelebA: %d' % args.attribute)

        logging.info('######################################')

        if args.dataset in ('mnist', 'fmnist','kdd99'): 
            dataset = load_dataset(args.dataset, data_path, args.normalclass, args.contamratio, random_seed)
        else:
            crop = CenterCrop(140)
            scale = Rescale(64)
            dataset = CelebA_Dataset(csv_file, root_dir , args.attribute, args.normalclass, train_ratio, test_ratio, val_ratio, args.contamratio, random_seed, transform=transforms.Compose([crop, scale, ToTensor()]))

        logging.info('##########Training Condition##########')
        logging.info('Computation device: %s' % device)
        logging.info('Training optimizer: %s' % optimizer_name)
        logging.info('Training learning rate: %g' % lr)
        logging.info('Training epochs: %d' % args.epoch)
        logging.info('Training batch size: %d' % batch_size)

        logging.info('############Other parameters###########')

        logging.info('Sigma: %g' % sigma)
        logging.info('MMD batch size: %d' % mmd_batch_size)
        logging.info('Z dimension: %d' % z_dim)

        logging.info('MMD Weight: %g' % args.mmdweight)
        logging.info('Lipschitz Penalty Weight: %g' % args.lipschitzweight)
        logging.info('Lipschitz constant: %g' % args.lipschitzconstant)

        logging.info('#######################################')

        # Class
        Lcae = LCAE(args.dataset, sigma, mmd_batch_size, z_dim, scale_list, c, args.lipschitzconstant, args.mmdweight, args.lipschitzweight, args.contamratio, random_seed, args.normalclass,args.attribute, directory)
        Lcae.set_network(net_name)
        if (args.dataset =='mnist') or (args.dataset =='celeba') or (args.dataset =='fmnist'):
            Lcae.set_decoder_network(net_name+'_Decoder')
        else:
            Lcae.set_decoder_network(net_name)
        
        # Train model on dataset
        Lcae.train(dataset, optimizer_name, lr, args.epoch, batch_size, device)
        # Test model
        performance = Lcae.test(dataset, device, args.tsne)
        with open('{}'.format(outcome_text_file), 'a+') as f:
            f.write('the results of seed {} : '.format(random_seed) + str(performance)+'\n')

        logging.info('GPU NUM: %s' % args.gpu)
        logging.info('###########Data Property#############')
        logging.info('Train ratio: %d' % train_ratio)
        logging.info('Test ratio: %d' % test_ratio)
        logging.info('Validation ratio: %d' % val_ratio)
        logging.info('Contamination ratio: %d' % args.contamratio)
        logging.info('Random seed: %d' % random_seed)
        if args.dataset != 'kdd99':
            logging.info('Normal class: %d' % args.normalclass)
            if args.dataset != 'celeba':
                logging.info('Attribute for CelebA: %d' % args.attribute)

        logging.info('######################################')

        logging.info('##########Training Condition##########')
        logging.info('Computation device: %s' % device)
        logging.info('Training optimizer: %s' % optimizer_name)
        logging.info('Training learning rate: %g' % lr)
        logging.info('Training epochs: %d' % args.epoch)
        logging.info('Training batch size: %d' % batch_size)

        logging.info('############Other parameters###########')

        logging.info('Sigma: %g' % sigma)
        logging.info('MMD batch size: %d' % mmd_batch_size)
        logging.info('Z dimension: %d' % z_dim)

        logging.info('MMD Weight: %g' % args.mmdweight)
        logging.info('Lipschitz Penalty Weight: %g' % args.lipschitzweight)
        logging.info('Lipschitz constant: %g' % args.lipschitzconstant)

        logging.info('#######################################')

        store_performance.append(performance)


    store_performance = np.reshape(store_performance, (-1, 7))
    
    print('[accuracy, specificity, precision, recall, f1, roc_auc, auprc]')
    logging.info("Mean Outcome of %s: " % log_name)
    logging.info(np.mean(store_performance, axis=0))
    logging.info("Minimum Outcome of %s: " % log_name)
    logging.info(np.min(store_performance, axis=0))
    logging.info("Std Outcome of %s: " % log_name)
    logging.info(np.std(store_performance, axis=0))
    
    summerize_performance(outcome_text_file, np.mean(store_performance, axis=0), np.min(store_performance, axis=0)
                          , np.std(store_performance, axis=0))       