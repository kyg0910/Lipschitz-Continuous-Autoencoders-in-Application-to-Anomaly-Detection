import argparse, importlib, json, logging, os, random, time, torch
import torchvision.transforms as transforms
import numpy as np 
from dataset.main import load_dataset
from dataset.celeba import CelebA_Dataset
from utils import *
from base import BaseADDataset
from other_methods.svdd.architecture import build_network
from other_methods.svdd.deepSVDD_trainer import DeepSVDDTrainer
import configuration

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, dataset_name, objective: str , nu: float ,contam_ratio: int,
                 random_seed: int, normal_class: int, attribute: int, saving_directory: str):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }
        self.dataset_name = dataset_name
        self.contam_ratio = contam_ratio
        self.random_seed = random_seed
        self.normal_class = normal_class
        self.attribute = attribute
        self.saving_directory = saving_directory

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def set_network(self, net_name):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str, lr: float, n_epochs: int, batch_size: int, device: str):
        """Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.dataset_name, self.objective, self.R, self.c, self.nu, optimizer_name, lr,
                                       n_epochs, batch_size,device,  self.contam_ratio, self.random_seed, self.normal_class, 
                                       self.attribute, self.saving_directory)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, device=device)

        output = self.trainer.test(dataset, self.net)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        return output


    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


def run(args):
    ################################################################################
    # Settings: logging and declare variables
    ################################################################################
    # for variables
    CONFIG_DICT = configuration.load_config(args.model,args.dataset)
    globals().update(CONFIG_DICT)
 
    device = 'cuda'
    
    directory = make_dir_to_save_results(args.model,args.dataset,args.contamratio)
    if args.dataset == 'kdd99':
        outcome_text_file = '{}/outcome_{}_{}_contam ratio={}_replications={}.txt'.format(directory,args.model,args.dataset,args.contamratio,args.number)
        log_name = 'SVDD' + '_' + '{}_contam ratio={}_'.format(args.dataset,args.contamratio) +'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    elif (args.dataset == 'mnist') or (args.dataset == 'fmnist') :
        outcome_text_file = '{}/outcome_{}_{}_contam ratio={}_normal class={}_replications={}.txt'.format(directory,args.model,args.dataset, args.contamratio,args.normalclass,args.number)
        log_name = 'SVDD' + '_' + '{}_contam ratio={}_normal class={}_'.format(args.dataset,args.contamratio,args.normalclass) +'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    elif args.dataset == 'celeba' :
        outcome_text_file = '{}/outcome_{}_{}_contam ratio={}_attribute={}_normal class={}_replications={}.txt'.format(directory,args.model,args.dataset, args.contamratio,
                                                                            args.attribute, args.normalclass,args.number)
        log_name = 'SVDD' + '_' + '{}_contam ratio={}_normal class={}_attribute={}_'.format(args.dataset,args.contamratio,args.normalclass,args.attribute) +'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    
    # For logging(to be fixed)
    if os.path.exists('../loggings/svdd/{}'.format(args.dataset)) is False:
        os.makedirs('../loggings/svdd/{}'.format(args.dataset))

    if os.path.exists('../loggings/svdd/{}/{}.log'.format(args.dataset, log_name)) is True:
        os.remove('../loggings/svdd/{}/{}.log'.format(args.dataset, log_name))
    logging.basicConfig(filename='../loggings/svdd/{}/{}.log'.format(args.dataset,log_name),
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
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        device = 'cuda'
        if not torch.cuda.is_available():
            device = 'cpu'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        ################################################################################
        # TRAIN
        ################################################################################
        # Load the data
        if not torch.cuda.is_available():
            device = 'cpu'
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
        logging.info('Z dimension: %d' % z_dim)
        logging.info('#######################################')

        # Class
        deepsvdd = DeepSVDD(args.dataset, objective, nu, args.contamratio,random_seed, 
                            args.normalclass,args.attribute,directory)
        deepsvdd.set_network(net_name)
        
        
        # Train model on dataset
        deepsvdd.train(dataset,optimizer_name, lr,args.epoch,batch_size,device)

        # Test model
        performance = deepsvdd.test(dataset, device)

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

        logging.info('Z dimension: %d' % z_dim)

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