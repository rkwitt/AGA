import torch
import torch.utils.data as data_utils
from torch.autograd import Variable

from models import models

import os
import sys
import h5py
import pickle
import logging
import argparse
import numpy as np
from termcolor import colored, cprint


def setup_parser():
    parser = argparse.ArgumentParser(description='Evaluation of phi')
    parser.add_argument(
        "--dim",
        type=int, 
        default=4096, 
        help="dimensionality of features (default: 4096)")
    parser.add_argument(
        "--data_file",          
        metavar='', 
        help="data file to use")
    parser.add_argument(
        "--object_file",          
        metavar='', 
        help="object class information")
    parser.add_argument(
        "--rho_model",          
        metavar='', 
        help="rho model file")
    parser.add_argument(
        "--phi_model",          
        metavar='', 
        help="phi model information")
    parser.add_argument(
        "--batch_size",         
        metavar='', 
        type=int, 
        default=300, 
        help="batch size for training (default: 300)")
    parser.add_argument(
        '--no_cuda',            
        action='store_true', 
        default=False, 
        help='disables CUDA training (default: False)')
    parser.add_argument(
        "--verbose",            
        action="store_true", 
        default=False, 
        dest="verbose", 
        help="enables verbose output")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.verbose:
        cprint('Using CUDA: %s' % bool(args.cuda), 'blue')

    with open(args.object_file, 'rb') as input:
        object_classes = pickle.load(input)

    with open(args.phi_model + '.pkl', 'rb') as input:
        phi_dict = pickle.load(input)

    with open(args.data_file, 'rb') as input:
        data = pickle.load(input)

    assert len(data) == len(phi_dict)

    rho_model = models.rho(args.dim)
    rho_model.load_state_dict(torch.load(args.rho_model))
    rho_model.eval()

    for key in phi_dict:

        for i, target in enumerate(phi_dict[key]['targets']):

            model_file = phi_dict[key]['model_files'][i]

            phi_model = models.phi(args.dim)
            phi_model.load_state_dict(torch.load(
                os.path.join(args.phi_model, model_file)
                ))
            phi_model.eval()

            N = data[key]['data_feat'].shape[0]
            X = torch.from_numpy(data[key]['data_feat'].astype(np.float32))
            y = torch.ones(N,1) * target

            X = Variable(X)
            y = Variable(y)

            val = rho_model(phi_model(X))
            cprint('Interval [{:.2f} {:.2f}] -> Target: {:.2f}: {:.4f}'.format(
                phi_dict[key]['interval'][0],
                phi_dict[key]['interval'][1],
                target,
                (val-target).abs().mean().cpu().data.numpy()[0]), 'blue')
            
if __name__ == "__main__":
    main()




    
