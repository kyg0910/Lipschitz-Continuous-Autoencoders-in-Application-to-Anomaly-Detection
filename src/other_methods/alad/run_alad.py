import argparse, importlib, logging, os, shutil, time, urllib3, zipfile
import numpy as np
from other_methods.alad import alad_trainer
from utils import *
import configuration

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run(args):
    CONFIG_DICT = configuration.load_config(args.model,args.dataset)
    globals().update(CONFIG_DICT)
    
    directory = make_dir_to_save_results(args.model, args.dataset, args.contamratio)
    if args.dataset == 'kdd99':
        outcome_text_file = '{}/outcome_{}_{}_contam ratio={}_replications={}.txt'.format(directory,args.model,args.dataset,args.contamratio,args.number)
        log_name = 'ALAD'+'_'+'{}_contam ratio={}_'.format(args.dataset,args.contamratio)+'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    elif (args.dataset == 'mnist') or (args.dataset == 'fmnist') :
        outcome_text_file = '{}/outcome_{}_{}_contam ratio={}_normal class={}_replications={}.txt'.format(directory,args.model,args.dataset, args.contamratio,args.normalclass,args.number)
        log_name = 'ALAD'+'_'+'{}_contam ratio={}_normal class={}_'.format(args.dataset,args.contamratio,args.normalclass)+'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    elif args.dataset == 'celeba' :
        outcome_text_file = '{}/outcome_{}_{}_contam ratio={}_attribute={}_normal class={}_replications={}.txt'.format(directory,args.model,args.dataset, args.contamratio, args.attribute, args.normalclass,args.number)
        log_name = 'ALAD'+'_'+'{}_contam ratio={}_normal class={}_attribute={}_'.format(args.dataset,args.contamratio,args.normalclass,args.attribute)+'timestamp='+'_'.join([str(s) for s in list(time.localtime(time.time())[1:6])])
    
    # For logging(to be fixed)
    if os.path.exists('../loggings/alad/{}'.format(args.dataset)) is False:
        os.makedirs('../loggings/alad/{}'.format(args.dataset))

    if os.path.exists('../loggings/alad/{}/{}.log'.format(args.dataset, log_name)) is True:
        os.remove('../loggings/alad/{}/{}.log'.format(args.dataset, log_name))
    logging.basicConfig(filename='../loggings/alad/{}/{}.log'.format(args.dataset,log_name),
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
        tf.set_random_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        
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
        
        performance = alad_trainer.run(args.dataset, args.epoch, args.degree, random_seed, args.normalclass, args.enable_dzz,args.enable_sm, args.m,args.enable_early_stop, args.sn, args.contamratio, args.gpu, args.attribute, directory)
        
        with open('{}'.format(outcome_text_file), 'a+') as f:
            f.write('the results of seed {} : '.format(random_seed) + str(performance)+'\n')
        
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
