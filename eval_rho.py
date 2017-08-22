"""Evaluation script for the attribute strength predictor function (rho).

Author: rkwitt, mdixit (2017)
"""

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils

from models import models

import pickle
import logging
from termcolor import colored, cprint
import argparse
import h5py
import numpy as np
import sys


def setup_parser():
    parser = argparse.ArgumentParser(description='Evaluation of attribute strength regressor')
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
        help="model to use")
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
    

def show_stats(info, objects):
    mae = info[:,0]
    cid = info[:,1] 
    u, counts = np.unique(cid,return_counts=True)

    for val in u:
        idx = np.where(cid==val)[0]
        cprint('{:20s}: {:.2f} [m]'.format(
            objects[int(val)], mae[idx].mean()), 'blue')

    cprint('{:20s}: {:.2f} [m]'.format(
            'AVERAGE', mae.mean()), 'blue')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.verbose:
        cprint('Using CUDA: %s' % bool(args.cuda), 'blue')

    with open(args.object_file, 'rb') as input:
        object_classes = pickle.load(input)

    with open(args.data_file) as fid:
        data = pickle.load(fid)

    data_attr = data['data_attr'].astype(np.float32).reshape((data['data_attr'].shape[0],1))
    data_feat = data['data_feat'].astype(np.float32)
    data_oidx = data['data_oidx']

    N_data_feat, D_data_feat = data_feat.shape
    N_data_attr, D_data_attr = data_attr.shape

    assert N_data_feat == N_data_attr 

    if args.verbose:
        cprint('Data: (%d x %d)' % (N_data_feat, D_data_feat), 'blue')
        cprint('Attribute target: (%d x %d)' % (N_data_attr, D_data_attr), 'blue' )

    data_feat_th = torch.from_numpy(data_feat)
    data_attr_th = torch.from_numpy(data_attr)

    eval = data_utils.TensorDataset(data_feat_th, data_attr_th)
    eval_loader = data_utils.DataLoader(
        eval, 
        batch_size=args.batch_size, 
        shuffle=False)

    model = models.rho(args.dim)
    model.load_state_dict(torch.load(args.rho_model))
    if args.cuda:
        model.cuda()
    model.eval()

    info = np.zeros((data_feat.shape[0],2))
    for i, (src, tgt) in enumerate(eval_loader):
    
        if args.cuda:
            src = src.cuda()
            tgt = tgt.cuda()

        src_var = Variable(src)
        tgt_var = Variable(tgt)

        out_var = model(src_var)
        tmp_mae = (out_var - tgt_var).abs().cpu().data.numpy()

        i0 = i*args.batch_size
        i1 = i*args.batch_size+tmp_mae.shape[0]

        info[i0:i1,0] = tmp_mae.ravel()
        info[i0:i1,1] = data_oidx[i0:i1]

    show_stats(info, object_classes)

if __name__ == "__main__":
    main()




    
