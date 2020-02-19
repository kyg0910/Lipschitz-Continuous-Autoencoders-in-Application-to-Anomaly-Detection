import logging, time, torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, precision_recall_fscore_support 
from torch.utils.data.dataloader import DataLoader
from base import BaseTrainer, BaseADDataset, BaseNet
from utils import *

class DeepSVDDTrainer(BaseTrainer):
    def __init__(self,dataset_name, objective, R, c, nu: float, optimizer_name: str, lr: float, n_epochs: int,
                  batch_size: int, device: str,  contam_ratio, random_seed, 
                 normal_class, attribute,saving_directory):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, device)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        
        # Deep SVDD parameters
        self.dataset_name = dataset_name
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.contam_ratio = contam_ratio
        self.random_seed = random_seed
        self.normal_class = normal_class
        self.n_epochs = n_epochs
        self.attribute = attribute
        self.saving_directory = saving_directory

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        self.network_parameter_filename = name_filename('SVDD','network_parameter', self.saving_directory, self.dataset_name, self.contam_ratio, self.random_seed,self.attribute, self.normal_class,self.n_epochs)
        
        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()
        logging.info('printing network architecture : {}'.format(net))

        # Set device for network
        logging.info(net)
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ , val_loader= dataset.loaders(batch_size=self.batch_size)

        # Set optimizer (Adam optimizer)
        weight_decay_coef = 1e-3
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=weight_decay_coef)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logging.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logging.info('Center c initialized.')

        # Training
        logging.info('Starting training...')
        start_time = time.time()
        net.train()
        patience = 0

        data_loaders = {"train": train_loader, "val": val_loader}
        for epoch in range(self.n_epochs):
            loss_epoch = 0.0
            n_batches = 0
            loss_val_epoch = 0.0
            n_val_batches = 0
            epoch_start_time = time.time()
            
            for phase in ['train', 'val']:
                for data in data_loaders[phase]:
                    inputs, _, _ = data
                    inputs = inputs.to(self.device)

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = net(inputs.float())
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        if phase == 'train':
                            loss = torch.mean(dist)
                        else:
                            l2_reg = None
                            for W in net.parameters():
                                if l2_reg is None:
                                    l2_reg = W.norm(2) ** 2
                                else:
                                    l2_reg = l2_reg + W.norm(2) ** 2
                            loss = torch.mean(dist) + weight_decay_coef * l2_reg

                    # Update hypersphere radius R on mini-batch distances
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                        self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_epoch += loss.item()
                        n_batches += 1
                    else:
                        loss_val_epoch += loss.item()
                        n_val_batches += 1
                    
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logging.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} Val_Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, loss_val_epoch / n_val_batches))
            if epoch == 0:
                previous = loss_val_epoch/ n_val_batches
                logging.info('best model with validation loss : {} - saving...'.format(previous))   
                torch.save(net.state_dict(), self.network_parameter_filename)
            elif (epoch != 0) & (previous > loss_val_epoch/ n_val_batches):
                previous = loss_val_epoch/ n_val_batches
                patience = 0
                logging.info('best model with validation loss : {} - saving...'.format(previous))
                torch.save(net.state_dict(), self.network_parameter_filename)
            else:
                patience += 1
                logging.info('best model with validation loss : {}'.format(previous))
                logging.info('patience for early stopping : {}'.format(patience))
            if patience == 10:
                break

        self.train_time = time.time() - start_time
        logging.info('Training time: %.3f' % self.train_time)
        logging.info('Average training time per one epoch : %.3f' % (self.train_time/(epoch+1)))                
        net.load_state_dict(torch.load(self.network_parameter_filename))

        logging.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader , _ = dataset.loaders(batch_size=self.batch_size)

        # Testing
        logging.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs.float())
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logging.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        if self.dataset_name =='kdd99':
            self.accuracy, self.specificity, self.precision, self.recall, self.f1, self.roc_auc, self.auprc = calculate_performance(scores, labels, np.percentile(scores, 80))
            performance = calculate_performance(scores, labels, np.percentile(scores, 80))
        else:
            self.accuracy, self.specificity, self.precision, self.recall, self.f1, self.roc_auc, self.auprc = calculate_performance(scores, labels, np.percentile(scores, 10))
            performance = calculate_performance(scores, labels, np.percentile(scores, 10))

        logging.info('Test set precision: {:.4f}%'.format(100. * self.precision))
        logging.info('Test set recall: {:.4f}%'.format(100. * self.recall))
        logging.info('Test set f1: {:.4f}%'.format(100. * self.f1))
        logging.info('Test set AUC: {:.4f}%'.format(100. * self.roc_auc))
        logging.info('Test set AUPRC: {:.4f}%'.format(100. * self.auprc))
        
        logging.info('Finished testing.')
        
        return performance

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs.float())
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
