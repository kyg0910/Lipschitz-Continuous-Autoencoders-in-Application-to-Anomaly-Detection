import argparse, importlib

def main(args):
    if args.model == 'lcae':
        model = importlib.import_module('run_lcae')
    else:
        model = importlib.import_module('other_methods.{}.run_{}'.format(args.model,args.model))
    model.run(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment for anomaly detection models.')
    
    #####DEFAULT######
    parser.add_argument('model', nargs="?", choices=['lcae', 'svdd', 'alad'], help='the model name of the example you want to run')
    parser.add_argument('dataset', nargs="?", choices=['kdd99', 'mnist', 'fmnist', 'celeba'], help='the name of the dataset you want to run the experiment on')
    parser.add_argument('-g', '--gpu', nargs="?", type=int, default=0, help='which gpu to use')
    parser.add_argument('-c', '--contamratio', nargs="?", type=int, default=0,  choices = [0,5], help='contaminated ratio in training set')
    parser.add_argument('-d', '--decide', nargs="?",choices=['int','list'], default='int',  help='number or list of experiment')    
    parser.add_argument('-n', '--number', nargs="?", default=10,  help='number or list of experiment')
    
    parser.add_argument('-e', '--epoch', nargs="?", type=int, default=50, help='number of epochs')
    parser.add_argument('-N', '--normalclass', nargs="?", type=int, default=0, help='normal class')
    
    #####FOR CELEBA : in this case --norm == -1 or 1
    parser.add_argument('-a', '--attribute', nargs="?", type=int, default=0, help='attribute class')
    
    #####LCAE#####
    parser.add_argument('-t', '--tsne', nargs="?", type=bool, default=False, help='tsne for original/reconstructed data')        
    parser.add_argument('-m', '--mmdweight', nargs="?", type=float, default=0.0, help='mmd loss weight')
    parser.add_argument('-l', '--lipschitzweight', nargs="?", type=float, default=0.0, help='lipschitz loss weight')
    parser.add_argument('-C', '--lipschitzconstant', nargs="?", type=float, default=0.95, help='lipschitz constant')
    
    #####ALAD#####
    parser.add_argument('--m', nargs="?", default='fm',  choices=['cross-e', 'fm'], help='mode/method for discriminator loss')
    parser.add_argument('--w', nargs="?", type=float, default=0.1, help='weight for ALAD')
    parser.add_argument('--degree', nargs="?", type=int, default=1, help='degree for the L norm')
    parser.add_argument('--enable_sm', default=True, action='store_true',  help='enable TF summaries')
    parser.add_argument('--enable_dzz', default=True, action='store_true', help='enable dzz discriminator')
    parser.add_argument('--enable_early_stop', default=True, action='store_true', help='enable early_stopping')
    parser.add_argument('--sn', action='store_true', default=True, help='enable spectral_norm')
    
    main(parser.parse_args())
