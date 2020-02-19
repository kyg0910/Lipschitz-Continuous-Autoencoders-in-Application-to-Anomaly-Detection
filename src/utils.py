import os, sklearn, torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support
from skimage import io, transform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def calculate_performance(score, y_true, threshold):
    ###############################################################
    # Calculating accuracy/specificity/precision/recall/precision #
    # using the given threshold and auc/auprc.                    #
    #                                                             #
    # [Input]                                                     #
    # score: float32 array with shape [n, ].                      #
    # y_true: float32 array with shape [n, ].                     #
    # threshold: float32 array with shape [1, ].                  #
    #                                                             #
    # [Output]                                                    #
    # list of accuracy/specificity/precision/recall/f1            #
    #         /roc_auc/auprc                                      #
    ###############################################################

    y_hat = (score > threshold)
    TP = np.sum((y_hat == 1) & (y_true == 1))
    TN = np.sum((y_hat == 0) & (y_true == 0))
    FP = np.sum((y_hat == 1) & (y_true == 0))
    FN = np.sum((y_hat == 0) & (y_true == 1))

    accuracy = np.true_divide((TP + TN), (TP + FP + TN + FN))
    specificity = np.true_divide((TN), (TN + FP))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_hat, average='binary')

    # Compute AUC
    fpr, tpr, _ = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(y_true, score)

    return [accuracy, specificity, precision, recall, f1, roc_auc, auprc]


def calculate_mmd(encoded, mmd_batch_size, z_dim, sigma, c, scale_list):
    z = np.random.normal(size=(mmd_batch_size, z_dim), loc=0.0, scale=sigma)

    norms_real_real = np.sum(np.square(z), axis=1)
    dotprods_real_real = np.matmul(z, np.transpose(z))
    l2_square_real_real = (np.tile(norms_real_real, (mmd_batch_size, 1))
                           + np.transpose(np.tile(norms_real_real, (mmd_batch_size, 1)))
                           - 2. * dotprods_real_real)

    norms_fake_fake = np.sum(np.square(encoded), axis=1)
    dotprods_fake_fake = np.matmul(encoded, np.transpose(encoded))
    l2_square_fake_fake = (np.tile(norms_fake_fake, (np.shape(encoded)[0], 1))
                           + np.transpose(np.tile(norms_fake_fake, (np.shape(encoded)[0], 1)))
                           - 2. * dotprods_fake_fake)

    dotprods_real_fake = np.matmul(z, np.transpose(encoded))
    l2_square_real_fake = (np.tile(norms_fake_fake, (mmd_batch_size, 1))
                           + np.transpose(np.tile(norms_real_real, (np.shape(encoded)[0], 1)))
                           - 2. * dotprods_real_fake)

    current_mmd = np.zeros(np.shape(encoded)[0])
    for scale in scale_list:
        current_c = c * scale
        res1 = current_c / (current_c + l2_square_real_real)
        res1 = np.mean(res1)
        
        res2 = current_c / (current_c + l2_square_fake_fake)
        res2 = np.diag(res2)

        res3 = current_c / (current_c + l2_square_real_fake)
        res3 = np.mean(res3, axis=0)

        current_mmd += res1 + res2 - 2.0 * res3
    return current_mmd

###Results related
def make_dir_to_save_results(model_name,dataset_name,contam_ratio, *hParams):
    hparam = str(hParams)[1:-1].replace(" ", "").replace(',', '_')
    if len(hparam) == 0:
        if os.path.exists('../results/{}/{}/contam ratio={}'.format(model_name,dataset_name, contam_ratio)) is False:
            os.makedirs('../results/{}/{}/contam ratio={}'.format(model_name,dataset_name, contam_ratio))
        saving_directory = '../results/{}/{}/contam ratio={}'.format(model_name,dataset_name, contam_ratio)
    else:
        if os.path.exists('../results/{}/{}/hparam={}/contam ratio={}'.format(model_name,dataset_name,hparam, contam_ratio)) is False:
            os.makedirs('../results/{}/{}/hparam={}/contam ratio={}'.format(model_name,dataset_name,hparam, contam_ratio))
        saving_directory = '../results/{}/{}/hparam={}/contam ratio={}'.format(model_name,dataset_name,hparam, contam_ratio)
    
    return saving_directory

def summerize_performance(outcome_text_file, outcome_mean, outcome_min, outcome_std):
    #outcome_mean, outcome_min, outcome_std : numpy array
    with open('{}'.format(outcome_text_file), 'a+') as f:
            f.write('the mean of performances from all seeds : ' + str(outcome_mean)+'\n')
            f.write('the minimum of performances from all seeds : ' + str(outcome_min)+'\n')
            f.write('the stdev of performances from all seeds : ' + str(outcome_std)+'\n')
    outcome_mean = np.around(outcome_mean*100, decimals=2)
    outcome_std = np.around(outcome_std*100, decimals=1)
    
    latex = str()
    for i in range(len(outcome_mean)):
        latex +=  '&' + str(outcome_mean[i])+'$\pm$'+ str(outcome_std[i]) + ' '  
    
    with open('{}'.format(outcome_text_file), 'a+') as f:        
        f.write('for latex :' +'\n')
        f.write(latex + '\n')
            
def name_filename(model, mode, directory, dataset,contam_ratio,random_seed,attribute,normal_class,n_epochs, *hParams):
    hparam = str(hParams)[1:-1].replace(" ", "").replace(',', '_')
    
    if os.path.exists('{}/{}'.format(directory,mode)) is False:
        os.makedirs('{}/{}'.format(directory,mode))
    
    if (len(hparam) == 0) and (model == 'ALAD'):
        
        if os.path.exists('{}/{}/normal class={}/random seed={}'.format(directory,mode,normal_class, random_seed)) is False:
            os.makedirs('{}/{}/normal class={}/random seed={}'.format(directory,mode,normal_class, random_seed))

        if dataset == 'kdd99':
            name = '{}/{}/normal class={}/random seed={}/{}_{}_{}_contam ratio={}_random seed={}_epochs={}'.format(directory, mode,normal_class, random_seed, model, dataset, mode, contam_ratio, random_seed, n_epochs)
        elif (dataset == 'mnist') or (dataset == 'fmnist') :
            name = '{}/{}/normal class={}/random seed={}/{}_{}_{}_contam ratio={}_random seed={}_normal class={}_epochs={}'.format(directory,mode, normal_class, random_seed, model, dataset, mode, contam_ratio, random_seed, normal_class, n_epochs)
        elif dataset == 'celeba':
            name = '{}/{}/normal class={}/random seed={}/{}_{}_{}_contam ratio={}_random seed={}_attribute={}_normal class={}_epochs={}'.format(directory, mode, normal_class, random_seed, model, dataset, mode,  contam_ratio, random_seed, attribute, normal_class,n_epochs)
    
    elif (len(hparam) == 0) and (model != 'ALAD'):

        if dataset == 'kdd99':
            name = '{}/{}/{}_{}_{}_contam ratio={}_random seed={}_epochs={}'.format(directory, mode,model, dataset,mode,contam_ratio,random_seed,n_epochs)
        elif (dataset == 'mnist') or (dataset == 'fmnist') :
            name = '{}/{}/{}_{}_{}_contam ratio={}_random seed={}_normal class={}_epochs={}'.format(directory,mode,model, dataset, mode,contam_ratio, random_seed, normal_class, n_epochs)
        elif dataset == 'celeba':
            name = '{}/{}/{}_{}_{}_contam ratio={}_random seed={}_attribute={}_normal class={}_epochs={}'.format(directory,mode,model, dataset, mode,  contam_ratio, random_seed, attribute, normal_class,n_epochs)
            
    else:
        if dataset == 'kdd99':
            name = '{}/{}/{}_{}_{}_hparam={}_contam ratio={}_random seed={}_epochs={}'.format(directory, mode,model, dataset,mode,hparam, contam_ratio,random_seed,n_epochs)
        elif (dataset == 'mnist') or (dataset == 'fmnist') :
            name = '{}/{}/{}_{}_{}_hparam={}_contam ratio={}_random seed={}_normal class={}_epochs={}'.format(directory,mode,model, dataset, mode,hparam, contam_ratio, random_seed, normal_class, n_epochs)
        elif dataset == 'celeba':
            name = '{}/{}/{}_{}_{}_hparam={}_contam ratio={}_random seed={}_attribute={}_normal class={}_epochs={}'.format(directory,mode,model, dataset, mode, hparam, contam_ratio, random_seed, attribute, normal_class,n_epochs)
    
    if mode == 'network_parameter':
        if model == 'ALAD':
            name = name+'.ckpt'
        else:
            name = name+'.pt'
    elif (mode == 'valid_img') or (mode == 'test_img') or (mode == 'test_generate_img') or (mode == 'tsne'):
        name = name+'.pdf'
        
    return name

def display_pic(inputs, mode, directory, n_epochs, save_img_every, column_length):
    if mode =='val_img':
        n_subpics = int(n_epochs / save_img_every) + 1

        fig = plt.figure(figsize=(n_subpics*1.5, 15))
        outer = gridspec.GridSpec(1, n_subpics, wspace=0.2, hspace=0.2)

        for i in range(n_subpics):
            inner = gridspec.GridSpecFromSubplotSpec(column_length, 1,
                            subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            for j in range(column_length):
                ax = plt.Subplot(fig, inner[j])
                if j == 0:
                    if i == 0:
                        ax.set_title('original')
                    else:    
                        ax.set_title('{} epochs'.format(save_img_every*i))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(inputs[column_length*i+j])
                fig.add_subplot(ax)
    elif mode =='test_img':
        n_subpics = 2

        fig = plt.figure(figsize=(n_subpics*1.5, 15))
        outer = gridspec.GridSpec(1, n_subpics, wspace=0.2, hspace=0.2)

        for i in range(n_subpics):
            inner = gridspec.GridSpecFromSubplotSpec(10, 1,
                            subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            for j in range(10):
                ax = plt.Subplot(fig, inner[j])
                if j == 0:
                    if i == 0:
                        ax.set_title('original')

                    else:    
                        ax.set_title('reconstruction')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(inputs[10*i+j])
                fig.add_subplot(ax)
    elif mode =='test_generate_img':
        fig = plt.figure(figsize=(1.5, 15))
        outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
        inner = gridspec.GridSpecFromSubplotSpec(10, 1,
                        subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        for j in range(10):
            ax = plt.Subplot(fig, inner[j])
            if j == 0:
                ax.set_title('generated')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(inputs[j])
            fig.add_subplot(ax)
        
 
    plt.savefig(directory)

###CelebA Dataset related: Rescale, CenterCrop, ToTensor###
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img

class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int(h/2) - int(new_h/2)
        left = int(w/2) - int(new_w/2)

        img = image[top: top + new_h,
                      left: left + new_w]
        
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.from_numpy(sample)

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()

def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale

    return x

#####ALAD dataset related

def batch_fill(testx, batch_size):
    """ Quick and dirty hack for filling smaller batch

    :param testx:
    :param batch_size:
    :return:
    """
    nr_batches_test = int(testx.shape[0] / batch_size)
    ran_from = nr_batches_test * batch_size
    ran_to = (nr_batches_test + 1) * batch_size
    size = testx[ran_from:ran_to].shape[0]
    new_shape = [batch_size - size]+list(testx.shape[1:])
    fill = np.ones(new_shape)
    return np.concatenate([testx[ran_from:ran_to], fill], axis=0), size

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

#####ALAD spectral norm related
def conv2d(inputs, filters, kernel_size, strides=1, padding='valid',
           use_bias=True, kernel_initializer=None,
           bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
           name=None,reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], filters], initializer=kernel_initializer,
                            regularizer=kernel_regularizer)
        bias = tf.get_variable("bias", [filters], initializer=bias_initializer)
        x = tf.nn.conv2d(input=inputs, filter=spectral_norm(w),
                         strides=[1, strides, strides, 1], padding=padding)
        if use_bias :
            x = tf.nn.bias_add(x, bias)
    return x

def dense(inputs, units, use_bias=True, kernel_initializer=None,
          bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
          name=None,reuse=None):

    with tf.variable_scope(name, reuse=reuse):
        inputs = tf.contrib.layers.flatten(inputs)
        shape = inputs.get_shape().as_list()
        channels = shape[-1]

        w = tf.get_variable("kernel", [channels, units], tf.float32,
                                 initializer=kernel_initializer, regularizer=kernel_regularizer)
        if use_bias :
            bias = tf.get_variable("bias", [units],
                                   initializer=bias_initializer)

            x = tf.matmul(inputs, spectral_norm(w)) + bias
        else :
            x = tf.matmul(inputs, spectral_norm(w))
    return x

def spectral_norm(w, iteration=1, eps=1e-12):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + eps)

        u_ = tf.matmul(v_hat, w)
        u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + eps)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

