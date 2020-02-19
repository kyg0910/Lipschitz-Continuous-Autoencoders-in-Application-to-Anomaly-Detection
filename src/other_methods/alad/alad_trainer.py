import importlib, logging, os, sys, time
import numpy as np
import random as rn
import tensorflow as tf
from skimage.transform import resize
from dataset.kdd99 import rescale_KDD99
from utils import *
from other_methods.alad.architecture import *

FREQ_PRINT = 200 # print frequency image tensorboard [20]
FREQ_EV = 1
PATIENCE = 10

def get_getter(ema):  
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, score_method, do_spectral_norm):
    """See parameters
    """
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Degree for L norms: ', degree)
    print('Normal class: ', label)
    print('Score method: ', score_method)
    print('Discriminator zz enabled: ', allow_zz)
    print('Spectral Norm enabled: ', do_spectral_norm)

def display_progression_epoch(j, id_max):
    """See epoch progression
    """
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def train_and_test(dataset, epochs, degree, random_seed, label,
                   allow_zz, enable_sm, score_method, enable_early_stop, do_spectral_norm, contam_ratio, attribute, saving_directory):
    """ Runs the AliCE on the specified dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        dataset (str): name of the dataset
        nb_epochs (int): number of epochs
        degree (int): degree of the norm in the feature matching
        random_seed (int): trying different seeds for averaging the results
        label (int): label which is normal for image experiments
        allow_zz (bool): allow the d_zz discriminator or not for ablation study
        enable_sm (bool): allow TF summaries for monitoring the training
        score_method (str): which metric to use for the ablation study
        enable_early_stop (bool): allow early stopping for determining the number of epochs
        do_spectral_norm (bool): allow spectral norm or not for ablation study
    """
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    # Import model and data
    architectures = {'kdd99'   : Kdd99_architecture,
                     'mnist'   : Mnist_architecture,
                     'fmnist'  : Fmnist_architecture,
                     'celeba'  : Celeba_architecture}
    network = architectures[dataset]()
    data = importlib.import_module("dataset.{}".format(dataset))

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.999
    
    network_parameter_filename = name_filename('ALAD','network_parameter', saving_directory, dataset, contam_ratio, random_seed, attribute, label, epochs)
    
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Placeholders
    if (dataset =='mnist') or (dataset =='fmnist'):
        x_pl = tf.placeholder(tf.float32, shape=(None, 32, 32, 1),
                              name="input_x")
    else:
        x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(),
                          name="input_x")
    z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim],
                          name="input_z")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")
    
    # Data
    logging.info('Data loading...')
    if dataset == 'kdd99':
        _, _, testx, testy, _, _, trainx, valx = rescale_KDD99(50, 25, 25, contam_ratio, random_seed)
    
    elif (dataset =='mnist') or (dataset =='fmnist'):
        trainx, trainy = data.get_train(label,True,True,contam_ratio,random_seed)
        testx, testy = data.get_test(label,True,True,contam_ratio,random_seed)
        valx, valy = data.get_valid(label,True,True,contam_ratio,random_seed)
        
        resized_trainx = np.zeros((np.shape(trainx)[0], 32, 32, 1))
        for i in range(np.shape(trainx)[0]):
            resized_trainx[i, :, :, 0] = resize(trainx[i, :, :], (32, 32))
        resized_valx = np.zeros((np.shape(valx)[0], 32, 32, 1))
        for i in range(np.shape(valx)[0]):
            resized_valx[i, :, :, 0] = resize(valx[i, :, :], (32, 32))
        resized_testx = np.zeros((np.shape(testx)[0], 32, 32, 1))
        for i in range(np.shape(testx)[0]):
            resized_testx[i, :, :, 0] = resize(testx[i, :, :], (32, 32))
        
        trainx, valx, testx = resized_trainx, resized_valx, resized_testx
    elif dataset == 'celeba':
        trainx, trainy, testx,testy,valx,valy = data.get_dataset(attribute, label, True, True, contam_ratio, random_seed)
    
    trainx_copy = trainx.copy()

    rng = np.random.RandomState(random_seed)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logging.info('Building graph...')

    logging.info("ALAD is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, score_method, do_spectral_norm)

    gen = network.decoder
    enc = network.encoder
    dis_xz = network.discriminator_xz
    dis_xx = network.discriminator_xx
    dis_zz = network.discriminator_zz

    with tf.variable_scope('encoder_model'):
        z_gen = enc(x_pl, is_training=is_training_pl,
                    do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('generator_model'):
        x_gen = gen(z_pl, is_training=is_training_pl)
        rec_x = gen(z_gen, is_training=is_training_pl, reuse=True)

    with tf.variable_scope('encoder_model'):
        rec_z = enc(x_gen, is_training=is_training_pl, reuse=True,
                    do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xz'):
        l_encoder, inter_layer_inp_xz = dis_xz(x_pl, z_gen,
                                            is_training=is_training_pl,
                    do_spectral_norm=do_spectral_norm)
        l_generator, inter_layer_rct_xz = dis_xz(x_gen, z_pl,
                                              is_training=is_training_pl,
                                              reuse=True,
                    do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xx'):
        x_logit_real, inter_layer_inp_xx = dis_xx(x_pl, x_pl,
                                                  is_training=is_training_pl,
                    do_spectral_norm=do_spectral_norm)
        x_logit_fake, inter_layer_rct_xx = dis_xx(x_pl, rec_x, is_training=is_training_pl,
                              reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_zz'):
        z_logit_real, _ = dis_zz(z_pl, z_pl, is_training=is_training_pl,
                                 do_spectral_norm=do_spectral_norm)
        z_logit_fake, _ = dis_zz(z_pl, rec_z, is_training=is_training_pl,
                              reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.name_scope('loss_functions'):
        # discriminator xz
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_generator),logits=l_generator))
        dis_loss_xz = loss_dis_gen + loss_dis_enc

        # discriminator xx
        x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.ones_like(x_logit_real))
        x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake))
        dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)

        # discriminator zz
        z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_real, labels=tf.ones_like(z_logit_real))
        z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake))
        dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)

        loss_discriminator = dis_loss_xz + dis_loss_xx + dis_loss_zz if \
            allow_zz else dis_loss_xz + dis_loss_xx

        # generator and encoder
        gen_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_generator),logits=l_generator))
        enc_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_encoder), logits=l_encoder))
        x_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.zeros_like(x_logit_real))
        x_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_fake, labels=tf.ones_like(x_logit_fake))
        z_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_real, labels=tf.zeros_like(z_logit_real))
        z_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.ones_like(z_logit_fake))

        cost_x = tf.reduce_mean(x_real_gen + x_fake_gen)
        cost_z = tf.reduce_mean(z_real_gen + z_fake_gen)

        cycle_consistency_loss = cost_x + cost_z if allow_zz else cost_x
        loss_generator = gen_loss_xz + cycle_consistency_loss
        loss_encoder = enc_loss_xz + cycle_consistency_loss

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dxzvars = [var for var in tvars if 'discriminator_model_xz' in var.name]
        dxxvars = [var for var in tvars if 'discriminator_model_xx' in var.name]
        dzzvars = [var for var in tvars if 'discriminator_model_zz' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis_xz = [x for x in update_ops if
                             ('discriminator_model_xz' in x.name)]
        update_ops_dis_xx = [x for x in update_ops if
                             ('discriminator_model_xx' in x.name)]
        update_ops_dis_zz = [x for x in update_ops if
                             ('discriminator_model_zz' in x.name)]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                  beta1=0.5)

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer.minimize(loss_generator, var_list=gvars,
                                            global_step=global_step)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer.minimize(loss_encoder, var_list=evars)

        with tf.control_dependencies(update_ops_dis_xz):
            dis_op_xz = optimizer.minimize(dis_loss_xz, var_list=dxzvars)

        with tf.control_dependencies(update_ops_dis_xx):
            dis_op_xx = optimizer.minimize(dis_loss_xx, var_list=dxxvars)

        with tf.control_dependencies(update_ops_dis_zz):
            dis_op_zz = optimizer.minimize(dis_loss_zz, var_list=dzzvars)

        # Exponential Moving Average for inference
        def train_op_with_ema_dependency(vars, op):
            ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op = ema.apply(vars)
            with tf.control_dependencies([op]):
                train_op = tf.group(maintain_averages_op)
            return train_op, ema

        train_gen_op, gen_ema = train_op_with_ema_dependency(gvars, gen_op)
        train_enc_op, enc_ema = train_op_with_ema_dependency(evars, enc_op)
        train_dis_op_xz, xz_ema = train_op_with_ema_dependency(dxzvars,
                                                               dis_op_xz)
        train_dis_op_xx, xx_ema = train_op_with_ema_dependency(dxxvars,
                                                               dis_op_xx)
        train_dis_op_zz, zz_ema = train_op_with_ema_dependency(dzzvars,
                                                               dis_op_zz)

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(x_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True,
                        do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('generator_model'):
        rec_x_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)
        x_gen_ema = gen(z_pl, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model_xx'):
        l_encoder_emaxx, inter_layer_inp_emaxx = dis_xx(x_pl, x_pl,
                                                    is_training=is_training_pl,
                                                    getter=get_getter(xx_ema),
                                                    reuse=True,
                    do_spectral_norm=do_spectral_norm)

        l_generator_emaxx, inter_layer_rct_emaxx = dis_xx(x_pl, rec_x_ema,
                                                      is_training=is_training_pl,
                                                      getter=get_getter(
                                                          xx_ema),
                                                      reuse=True,
                    do_spectral_norm=do_spectral_norm)

    with tf.name_scope('Testing'):
        with tf.variable_scope('Scores'):
            score_ch = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_emaxx),
                    logits=l_generator_emaxx)
            score_ch = tf.squeeze(score_ch)

            rec = x_pl - rec_x_ema
            rec = tf.contrib.layers.flatten(rec)
            score_l1 = tf.norm(rec, ord=1, axis=1,
                            keep_dims=False, name='d_loss')
            score_l1 = tf.squeeze(score_l1)

            rec = x_pl - rec_x_ema
            rec = tf.contrib.layers.flatten(rec)
            score_l2 = tf.norm(rec, ord=2, axis=1,
                            keep_dims=False, name='d_loss')
            score_l2 = tf.squeeze(score_l2)

            inter_layer_inp, inter_layer_rct = inter_layer_inp_emaxx, \
                                               inter_layer_rct_emaxx
            fm = inter_layer_inp - inter_layer_rct
            fm = tf.contrib.layers.flatten(fm)
            score_fm = tf.norm(fm, ord=degree, axis=1,
                             keep_dims=False, name='d_loss')
            score_fm = tf.squeeze(score_fm)

    if enable_early_stop:
        rec_error_valid = tf.reduce_mean(score_fm)

    if enable_sm:
        with tf.name_scope('summary'):
            with tf.name_scope('dis_summary'):
                tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
                tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
                tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])
                tf.summary.scalar('loss_dis_xz', dis_loss_xz, ['dis'])
                tf.summary.scalar('loss_dis_xx', dis_loss_xx, ['dis'])
                if allow_zz:
                    tf.summary.scalar('loss_dis_zz', dis_loss_zz, ['dis'])

            with tf.name_scope('gen_summary'):
                tf.summary.scalar('loss_generator', loss_generator, ['gen'])
                tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])
                tf.summary.scalar('loss_encgen_dxx', cost_x, ['gen'])
                if allow_zz:
                    tf.summary.scalar('loss_encgen_dzz', cost_z, ['gen'])

            if enable_early_stop:        
                with tf.name_scope('validation_summary'):
                    tf.summary.scalar('valid', rec_error_valid, ['v'])

            with tf.name_scope('img_summary'):
                heatmap_pl_latent = tf.placeholder(tf.float32,
                                                   shape=(1, 480, 640, 3),
                                                   name="heatmap_pl_latent")
                sum_op_latent = tf.summary.image('heatmap_latent', heatmap_pl_latent)
        
            
            heatmap_pl_rec = tf.placeholder(tf.float32, shape=(1, 480, 640, 3),
                                        name="heatmap_pl_rec")
            with tf.name_scope('image_summary'):
                tf.summary.image('heatmap_rec', heatmap_pl_rec, 1, ['image'])

            sum_op_dis = tf.summary.merge_all('dis')
            sum_op_gen = tf.summary.merge_all('gen')
            sum_op = tf.summary.merge([sum_op_dis, sum_op_gen])
            sum_op_im = tf.summary.merge_all('image')
            sum_op_valid = tf.summary.merge_all('v')

    saver = tf.train.Saver(max_to_keep=2)
    save_model_secs = None if enable_early_stop else 20 
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    init=tf.global_variables_initializer()
    logging.info('Start training...')
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                          log_device_placement=True, gpu_options = gpu_options)) as sess:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rn.seed(random_seed)
        
        sess.run(init)
        step = sess.run(global_step)
        logging.info('Initialization done at step {}'.format(step/nr_batches_train))
        train_batch = 0
        epoch = 0
        best_valid_loss = 0
        request_stop = False

        while epoch < epochs:
            lr = starting_lr
            begin = time.time()

             # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis_xz, train_loss_dis_xx,  train_loss_dis_zz, \
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0, 0, 0, 0]

            # Training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {x_pl: trainx[ran_from:ran_to],
                             z_pl: np.random.normal(size=[batch_size, latent_dim]),
                             is_training_pl: True,
                             learning_rate:lr}

                _, _, _, ld, ldxz, ldxx, ldzz, step = sess.run([train_dis_op_xz,
                                                              train_dis_op_xx,
                                                              train_dis_op_zz,
                                                              loss_discriminator,
                                                              dis_loss_xz,
                                                              dis_loss_xx,
                                                              dis_loss_zz,
                                                              global_step],
                                                             feed_dict=feed_dict)
                train_loss_dis += ld
                train_loss_dis_xz += ldxz
                train_loss_dis_xx += ldxx
                train_loss_dis_zz += ldzz

                # train generator and encoder
                feed_dict = {x_pl: trainx_copy[ran_from:ran_to],
                             z_pl: np.random.normal(size=[batch_size, latent_dim]),
                             is_training_pl: True,
                             learning_rate:lr}
                _,_, le, lg = sess.run([train_gen_op,
                                            train_enc_op,
                                            loss_encoder,
                                            loss_generator],
                                           feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le

                if enable_sm:
                    sm = sess.run(sum_op, feed_dict=feed_dict)
                    
                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train
            train_loss_dis_xz /= nr_batches_train
            train_loss_dis_xx /= nr_batches_train
            train_loss_dis_zz /= nr_batches_train

            logging.info('Epoch terminated')
            if allow_zz:
                logging.info("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
                      "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | "
                      "loss dis zz = %.4f"
                      % (epoch, time.time() - begin, train_loss_gen,
                         train_loss_enc, train_loss_dis, train_loss_dis_xz,
                         train_loss_dis_xx, train_loss_dis_zz))
            else:
                logging.info("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
                      "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | "
                      % (epoch, time.time() - begin, train_loss_gen,
                         train_loss_enc, train_loss_dis, train_loss_dis_xz,
                         train_loss_dis_xx))

            ##EARLY STOPPING
            if (epoch + 1) % FREQ_EV == 0 and enable_early_stop:
                if dataset != 'kdd99':
                    
                    valid_loss = 0
                    idx = np.array_split(np.arange(valx.shape[0]), int(valx.shape[0]/batch_size))
                    for i in range(len(idx)):
                        feed_dict = {x_pl: valx[idx[i], :, :, :],
                                     is_training_pl: False}
                        vl = sess.run(rec_error_valid, feed_dict=feed_dict)
                        valid_loss += vl
                    valid_loss /= len(idx)
                else:
                    valid_loss = 0
                    idx = np.array_split(np.arange(valx.shape[0]), int(valx.shape[0]/batch_size))
                    for i in range(len(idx)):
                        feed_dict = {x_pl: valx[idx[i], :],
                                     is_training_pl: False}
                        vl = sess.run(rec_error_valid, feed_dict=feed_dict)
                        valid_loss += vl
                    valid_loss /= len(idx)
                    
                if enable_sm:
                    sm = sess.run(sum_op_valid, feed_dict=feed_dict)

                logging.info('Validation: valid loss {:.4f}'.format(valid_loss))

                if (valid_loss < best_valid_loss or epoch == FREQ_EV-1):
                    best_valid_loss = valid_loss
                    logging.info("Best model - valid loss = {:.4f} - saving...".format(best_valid_loss))
                    saver.save(sess, network_parameter_filename, global_step=step)
                    best_step = step
                   
                    nb_without_improvements = 0
                else:
                    nb_without_improvements += FREQ_EV

                if nb_without_improvements > PATIENCE:
                    logging.info(
                      "Early stopping at epoch {} with weights from epoch {}".format(
                          epoch, epoch - nb_without_improvements))
                    break

            epoch += 1
        train_time = time.time()-start_time
        logging.info('Training time: %.3f' % train_time)
        logging.info('Average training time per one epoch : %.3f' % (train_time/(epoch+1)))
        logging.info('Testing evaluation...')
        
        SAVER_DIR = "../results/alad/{}/contam ratio={}/network_parameter/normal class={}/random seed={}".format(dataset, contam_ratio, label, random_seed)
        
        saver = tf.train.Saver()
        checkpoint_path = SAVER_DIR
        ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
        
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        scores_fm = []
        inference_time = []
        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_test_time_batch = time.time()

            feed_dict = {x_pl: testx[ran_from:ran_to],
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl:False}

            scores_fm += sess.run(score_fm, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_test_time_batch)

        inference_time = np.sum(inference_time)
        mean_inference_time = np.mean(inference_time)
        logging.info('Testing : inference time is %.4f' % (inference_time))
        logging.info('Testing : mean inference time is %.4f' % (mean_inference_time))

        if testx.shape[0] % batch_size != 0:
            batch, size = batch_fill(testx, batch_size)
            feed_dict = {x_pl: batch,
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl: False}
            bscores_fm = sess.run(score_fm,feed_dict=feed_dict).tolist()
            scores_fm += bscores_fm[:size]
        testy = testy.astype(int)
        
        logging.info('Result for scores_fm')
        
        if dataset == 'kdd99':
            accuracy, specificity, precision, recall, f1, roc_auc, auprc = calculate_performance(scores_fm, testy, np.percentile(scores_fm, 80))
        elif (dataset =='cifar10') or (dataset =='mnist') or (dataset =='fmnist'):
            accuracy, specificity, precision, recall, f1, roc_auc, auprc = calculate_performance(scores_fm, testy, np.percentile(scores_fm, 10))
        elif (dataset =='celeba'):
            accuracy, specificity, precision, recall, f1, roc_auc, auprc = calculate_performance(scores_fm, testy, np.percentile(scores_fm, 10))
            
        logging.info('Test set precision: {:.4f}%'.format(100. * precision))
        logging.info('Test set recall: {:.4f}%'.format(100. * recall))
        logging.info('Test set f1: {:.4f}%'.format(100. * f1))
        logging.info('Test set AUC: {:.4f}%'.format(100. * roc_auc))
        logging.info('Test set AUPRC: {:.4f}%'.format(100. * auprc))

        if dataset == 'kdd99':
            performance = calculate_performance(scores_fm, testy, np.percentile(scores_fm, 80))
        elif (dataset =='cifar10') or (dataset =='mnist') or (dataset =='fmnist'):
            performance = calculate_performance(scores_fm, testy, np.percentile(scores_fm, 10))
        elif (dataset =='celeba'):
            performance = calculate_performance(scores_fm, testy, np.percentile(scores_fm, 10))
         
        return performance
    
def run(dataset_name, n_epochs, degree, random_seed, normal_class, allow_zz, enable_sm, score_method, enable_early_stop, do_spectral_norm, contam_ratio, gpu, attribute, saving_directory):
    """ Runs the training process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        performance = train_and_test(dataset_name, n_epochs, degree, random_seed, normal_class, allow_zz, enable_sm, score_method, enable_early_stop,do_spectral_norm, contam_ratio, attribute, saving_directory)
    return performance
