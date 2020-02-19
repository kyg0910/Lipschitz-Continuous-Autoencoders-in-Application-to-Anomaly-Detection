from __future__ import print_function, division
import logging, os, time, torch
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from skimage import io, transform
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from base import BaseTrainer, BaseADDataset, BaseNet
from utils import *

class LCAETrainer(BaseTrainer):

    def __init__(self, dataset_name, optimizer_name, lr, n_epochs, batch_size, device, sigma, mmd_batch_size, z_dim, 
                 scale_list , c ,lipschitz_constant,  mmd_weight, lipschitz_penalty_weight, contam_ratio, random_seed, 
                 normal_class, attribute, saving_directory):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, device)
        
        self.dataset_name = dataset_name
        self.sigma = sigma
        self.mmd_batch_size = mmd_batch_size
        self.z_dim = z_dim
        self.scale_list =scale_list
        self.c = c
        self.lipschitz_constant = lipschitz_constant
        self.mmd_weight = mmd_weight
        self.lipschitz_penalty_weight = lipschitz_penalty_weight
        self.contam_ratio = contam_ratio
        self.random_seed = random_seed
        self.normal_class = normal_class
        self.n_epochs = n_epochs
        
        self.attribute = attribute
        self.saving_directory = saving_directory
        self.network_parameter_filename = name_filename('LCAE','network_parameter', self.saving_directory, self.dataset_name, self.contam_ratio, self.random_seed, self.attribute, self.normal_class,self.n_epochs, self.mmd_weight, self.lipschitz_penalty_weight, self.lipschitz_constant)
        if dataset_name != 'kdd99':
            self.valid_img_filename = name_filename('LCAE','valid_img', self.saving_directory, self.dataset_name, self.contam_ratio, self.random_seed, self.attribute, self.normal_class,self.n_epochs,  self.mmd_weight, self.lipschitz_penalty_weight,self.lipschitz_constant)
            self.test_img_filename = name_filename('LCAE','test_img', self.saving_directory, self.dataset_name, self.contam_ratio, self.random_seed, self.attribute, self.normal_class,self.n_epochs,  self.mmd_weight, self.lipschitz_penalty_weight,self.lipschitz_constant)
            self.test_generate_img_filename = name_filename('LCAE','test_generate_img', self.saving_directory, self.dataset_name, self.contam_ratio, self.random_seed, self.attribute, self.normal_class,self.n_epochs,  self.mmd_weight, self.lipschitz_penalty_weight,self.lipschitz_constant)
        
        # Saving decoder dictionary
        self.net_dict = None
        self.net_decoder_dict = None
        
        # Optimization parameters
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
    
    def train(self, dataset: BaseADDataset, net: BaseNet):        
        logger = logging.getLogger()

        # Set device for network
        logging.info('printing network architecture : {}'.format(net))
        net = net.to(self.device)

        # Get train data loader
        train_loader, _, val_loader = dataset.loaders(batch_size=self.batch_size)
        
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)

        # Training
        logging.info('Starting training...')
        start_time = time.time()
        net.train()
        patience = 0
        save_img_every = 5

        data_loaders = {"train": train_loader, "val": val_loader}
        for epoch in range(self.n_epochs):
            val_reconstruction_error = 0.0
            val_mmd = 0.0
            val_lipschitz_loss = 0.0
            
            loss_epoch = 0.0
            loss_val_epoch = 0.0
            recon_loss_epoch = 0.0
            recon_loss_val_epoch = 0.0
            mmd_loss_epoch = 0.0
            mmd_loss_val_epoch = 0.0
            lips_loss_epoch = 0.0
            lips_loss_val_epoch = 0.0
            
            n_batches = 0
            n_val_batches = 0
            epoch_start_time = time.time()
            
            for phase in ['train', 'val']:
                for data in data_loaders[phase]:
                    inputs, labels, idx = data
                    inputs = inputs.to(self.device)
                    
                    # Zero the network parameter gradients
                    optimizer.zero_grad()                    
                    encoded, reconstructed = net(inputs.float())
                    
                    # define reconstruction error
                    reconstruction_error = torch.mean(torch.sum((reconstructed-inputs.float())**2, dim=1))
                    
                    z_generator = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([self.sigma]))
                    z = z_generator.sample((self.mmd_batch_size, self.z_dim))
                    z = torch.reshape(z,(self.mmd_batch_size, self.z_dim))
                    z = z.to(self.device)
                    
                    # define mmd loss
                    mmd_norms_real_real = torch.sum(z**2, dim=1, keepdim=True)
                    mmd_dotprods_real_real = torch.mm(z, z.t())
                    mmd_l2_square_real_real = mmd_norms_real_real + mmd_norms_real_real.t() - 2. * mmd_dotprods_real_real

                    mmd_norms_fake_fake = torch.sum(encoded**2, dim=1, keepdim=True)
                    mmd_dotprods_fake_fake = torch.mm(encoded, encoded.t())
                    mmd_l2_square_fake_fake = mmd_norms_fake_fake + mmd_norms_fake_fake.t() - 2. * mmd_dotprods_fake_fake

                    mmd_dotprods_real_fake = torch.mm(z, encoded.t())
                    mmd_l2_square_real_fake = mmd_norms_real_real + mmd_norms_fake_fake.t() - 2. * mmd_dotprods_real_fake

                    mmd = 0.0
                    for scale in self.scale_list:
                        current_c = self.c * scale
                        res1 = current_c / (current_c + mmd_l2_square_fake_fake)
                        res1 = torch.mean(res1)

                        res2 = current_c / (current_c + mmd_l2_square_real_real)
                        res2 = torch.mean(res2)

                        res3 = current_c / (current_c + mmd_l2_square_real_fake)
                        res3 = torch.mean(res3)
                        
                        mmd += torch.mean(res1 + res2 - 2.0 * res3)

                    # define penalty term about Lipschitz continuity
                    reshaped_inputs = inputs.view(inputs.size(0), -1)
                    lipschitz_norms_real_real = torch.sum(reshaped_inputs.float()**2, dim=1, keepdim=True).to(self.device)
                    lipschitz_dotprods_real_real = torch.mm(reshaped_inputs.float(), reshaped_inputs.float().t()).to(self.device)
                    lipschitz_distances_real_real = torch.sqrt(torch.max(lipschitz_norms_real_real
                                                                         + lipschitz_norms_real_real.t()- 2. * lipschitz_dotprods_real_real, 1e-4*torch.ones((inputs.size()[0],1)).to(self.device)))                              
                    
                    reshaped_reconstructed = reconstructed.view(reconstructed.size(0), -1)
                    lipschitz_norms_fake_fake = torch.sum(reshaped_reconstructed**2, dim=1, keepdim=True)
                    lipschitz_dotprods_fake_fake = torch.mm(reshaped_reconstructed, reshaped_reconstructed.t()).to(self.device)
                    lipschitz_distances_fake_fake = torch.sqrt(torch.max(lipschitz_norms_fake_fake + lipschitz_norms_fake_fake.t()- 2. * lipschitz_dotprods_fake_fake, self.lipschitz_constant * 1e-4 *torch.ones((inputs.size()[0],1)).to(self.device)))
                    
                    lipschitz_loss = torch.mean(torch.max(lipschitz_distances_fake_fake / lipschitz_distances_real_real - self.lipschitz_constant*torch.ones((inputs.size()[0],1)).to(self.device), 0.0*torch.ones((inputs.size()[0],1)).to(self.device))) 
                    
                    # define total loss
                    loss = reconstruction_error +self.mmd_weight*mmd + self.lipschitz_penalty_weight*lipschitz_loss
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_epoch += loss.item()
                        
                        recon_loss_epoch += reconstruction_error.item()

                        mmd_loss_epoch += mmd.item()
                        lips_loss_epoch += lipschitz_loss.item()
                        n_batches += 1
                    else:
                        loss_val_epoch += loss.item()
                        
                        recon_loss_val_epoch += reconstruction_error.item()
                        mmd_loss_val_epoch += mmd.item()
                        lips_loss_val_epoch += lipschitz_loss.item()
                        n_val_batches += 1
                        
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logging.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}\t  Recon_loss: {:.8f}\t mmd_loss:  {:.8f}\t lipschitz_loss:  {:.8f}\t'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, recon_loss_epoch / n_batches, mmd_loss_epoch / n_batches, lips_loss_epoch / n_batches))
            
            logging.info('                               val_Loss: {:.8f}\t  val_Recon_loss: {:.8f}\t val_mmd_loss:  {:.8f}\t val_lipschitz_loss:  {:.8f}\t'
                        .format(loss_val_epoch / n_val_batches, recon_loss_val_epoch / n_val_batches, mmd_loss_val_epoch / n_val_batches, lips_loss_val_epoch / n_val_batches))

            if epoch == 0:
                previous = loss_val_epoch / n_val_batches
                logging.info('best model with validation loss : {} - saving...'.format(previous))
                torch.save(net.state_dict(), self.network_parameter_filename)
            
            elif (epoch != 0) & (previous > loss_val_epoch/ n_val_batches):
                previous = loss_val_epoch / n_val_batches
                patience = 0
                logging.info('best model with validation loss : {} - saving...'.format(previous))
                torch.save(net.state_dict(), self.network_parameter_filename)
            else:
                patience += 1
                logging.info('best model with validation loss : {}'.format(previous))
                logging.info('patience for early stopping : {}'.format(patience))

            if patience == 10:
                break
                
            if self.dataset_name != 'kdd99':
                if epoch == 0:
                    input_length = len(inputs[:10])
                    img_list = inputs[:10].detach().cpu().data.numpy()

                if epoch % save_img_every == 0:
                    with torch.no_grad():
                        net.eval()
                        _, reconstructed = net(inputs[:10].float())
                        reconstructed = reconstructed.detach().cpu().data.numpy()
                    img_list = np.append(img_list, reconstructed, axis = 0)
                    logging.info('shape of image:{}'.format(img_list.shape))
                    
        self.train_time = time.time() - start_time
        logging.info('Training time: %.3f' % self.train_time)
        logging.info('Average training time per one epoch : %.3f' % (self.train_time/(epoch+1)))
        
        if self.dataset_name != 'kdd99':
            if np.transpose(img_list, (0,2,3,1)).shape[3] == 3:
                img_list = np.transpose(img_list, (0,2,3,1))
            else:
                img_list = np.transpose(img_list, (0,2,3,1)).reshape(-1,32,32)
            display_pic(img_list, 'val_img', self.valid_img_filename, epoch, save_img_every, input_length)

        logging.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, net_decoder: BaseNet, show_tsne = False):
        if show_tsne:
            self.tsne_filename =name_filename('tsne', self.saving_directory, self.dataset_name, self.mmd_weight, self.lipschitz_penalty_weight,self.lipschitz_constant, self.contam_ratio, self.random_seed, self.attribute, self.normal_class,self.n_epochs)
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger = logging.getLogger()
        
        # Set device for network
        if self.dataset_name != 'kdd99':
            net_dict = net.state_dict()
            net_decoder_dict = net_decoder.state_dict()
            net_decoder_dict = {k: v for k, v in net_dict.items() if k in net_decoder_dict}
            net_decoder.load_state_dict(net_decoder_dict)
            net_decoder = net_decoder.to(self.device)
            net_decoder.eval()

        net = net.to(self.device)

        # Get test data loader
        _, test_loader, _ = dataset.loaders(batch_size=self.batch_size)

        # Testing
        logging.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        img_list = []
        img_list_generate = []
        recon = []
        origin = []
        net.eval()

        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                
                _, reconstructed = net(inputs.float())
                reconstructed = reconstructed.detach().cpu()
                
                encoded, reconstructed = net(inputs.float())
                reshaped_inputs = inputs.view(inputs.size(0), -1)
                reshaped_reconstructed = reconstructed.view(reconstructed.size(0), -1)
                
                # define reconstruction error
                reconstruction_error = torch.mean((reshaped_reconstructed-reshaped_inputs.float())**2, dim=1)
                reconstruction_error = reconstruction_error.cpu().data.numpy()
               
                encoded = encoded.cpu().data.numpy()
                mmd = np.zeros(np.shape(encoded)[0])
                mmd = calculate_mmd(encoded, self.mmd_batch_size, self.z_dim, self.sigma, self.c, self.scale_list)
                
                # define total loss
                scores = reconstruction_error
                
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.tolist())) 
                
                recon += reshaped_reconstructed.cpu().data.numpy().tolist()
                origin += inputs.view(len(inputs),-1).cpu().data.numpy().tolist()
                
        self.test_time = time.time() - start_time
            
        if self.dataset_name != 'kdd99':
            z_generator = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([self.sigma]))
            z = z_generator.sample((10, self.z_dim))
            z = torch.reshape(z,(10, self.z_dim))
            z = z.to(self.device)
            generated = net_decoder(z.float()).detach().cpu()
            
            
            img_list = np.append(inputs[:10].detach().cpu().data.numpy(),reconstructed[:10].detach().cpu().data.numpy()
                                 , axis = 0)
            img_list_generate = generated[:10].data.numpy()
            
            if np.transpose(img_list, (0,2,3,1)).shape[3] == 3:
                img_list = np.transpose(img_list, (0,2,3,1))
                img_list_generate = np.transpose(img_list_generate, (0,2,3,1))
            else:
                img_list = np.transpose(img_list, (0,2,3,1)).reshape(-1,32,32)
                img_list_generate = np.transpose(img_list_generate, (0,2,3,1)).reshape(-1,32,32)
          
            display_pic(img_list, 'test_img', self.test_img_filename, self.n_epochs, None, None)
            display_pic(img_list_generate, 'test_generate_img', self.test_generate_img_filename, self.n_epochs, None, None)

        logging.info('Testing time: %.3f' % self.test_time)
        
        if show_tsne:
            logging.info('T-sne for test data')

            _,labels,_ = zip(*idx_label_score)
            n_labels = len(labels)
            x_test = np.append(origin, recon, axis = 0)

            time_start = time.time()
            tsne = TSNE(n_components=2, verbose=1, n_iter=1000)
            tsne_results = tsne.fit_transform(x_test)
            print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
            SNE1= pd.DataFrame(data = tsne_results[:n_labels,:], columns = ['x', 'y'])
            SNE1['target'] = labels
            names = ['normal','abnormal']
            plt.clf()
            plt.figure(figsize = (10, 10))
            markers = ['o', 's']
            
            plt.xlabel('(a)',fontsize = 20)
            plt.xlim(-100, 100)
            plt.ylim(-100, 100)
            for i in range(len(names)):
                bucket = SNE1[SNE1['target'] == i]
                bucket = bucket.iloc[:,[0,1]].values
                plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i],alpha = 0.2) 
            plt.legend(loc='upper left',fontsize=12)
            plt.savefig(self.tsne_filename)

            logging.info('T-sne for test reconstruction data')
            time_start = time.time()
            SNE= pd.DataFrame(data = tsne_results[n_labels:,:], columns = ['x', 'y'])
            SNE['target'] = labels
            names = ['normal','abnormal']
            plt.clf()
            plt.figure(figsize = (10, 10))
            markers = ['o', 's']
            plt.xlabel('(b)',fontsize = 20)
            plt.xlim(-100, 100)
            plt.ylim(-100, 100)
            for i in range(len(names)):
                bucket = SNE[SNE['target'] == i]
                bucket = bucket.iloc[:,[0,1]].values
                plt.scatter(bucket[:, 0], bucket[:, 1], label=names[i],alpha = 0.2) 
            plt.legend(loc='upper left', fontsize=12)
            plt.savefig(self.tsne_filename)

        # Compute performances
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        if self.dataset_name == 'kdd99':
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