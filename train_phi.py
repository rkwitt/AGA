"""Training of the encoder-decoder part (phi).

Author: rkwitt, mdixit (2017)
"""

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils

from termcolor import colored, cprint

from models import models

import os
import uuid
import pickle
import logging
import argparse
import h5py
import numpy as np
import sys

class AGAEncoderDecoder(torch.nn.Module):

    def __init__(self):
        super(AGAEncoderDecoder, self).__init__()
        self.mod_phi = None
        self.mod_rho = None

    def set_rho(self, from_file, dim=4096):
        self.mod_rho = models.rho(dim)
        self.mod_rho.load_state_dict(
            torch.load(from_file))

    def get_rho(self):
        return self.mod_rho

    def set_phi(self, from_file, dim=4096):
        self.mod_phi = models.phi(dim)
        self.mod_phi.load_state_dict(
            torch.load(from_file))
        
    def get_phi(self):
        return self.mod_phi

    def forward(self, x):
        concat = [
            self.mod_phi(x), 
            self.mod_rho(self.mod_phi(x))]
        return concat


def setup_parser():
    parser = argparse.ArgumentParser(description="Train encoder-decoder part.")
    parser.add_argument(
        "--data_file",          
        metavar='', 
        help="data file to use")
    parser.add_argument(
        "--pretrained_phi",        
        metavar='', 
        help='pretrained model for phi')
    parser.add_argument(
        "--pretrained_rho",        
        metavar='', 
        help='pretrained model for rho')
    parser.add_argument(
        "--save",        
        metavar='', 
        help='name of output file')
    parser.add_argument(
        "--dim",
        type=int, 
        default=4096, 
        help="dimensionality of features (default: 4096)")
    parser.add_argument(
        "--verbose",            
        action="store_true", 
        default=False, 
        dest="verbose", 
        help="verbose output")
    parser.add_argument(
        "--batch_size",         
        metavar='', 
        type=int, 
        default=64, 
        help="batch size for training (default: 300)")
    parser.add_argument(
        "--learning_rate",      
        metavar='', 
        type=float, 
        default=1e-3, 
        help="learning rate (default: 0.001)")
    parser.add_argument(
        "--epochs",             
        metavar='', 
        type=int, 
        default=50, 
        help="#epochs to train (default: 50")
    parser.add_argument(
        '--no_cuda',            
        action='store_true', 
        default=False, 
        help='disables CUDA training (default: False)')
    return parser


def adjust_learning_rate(optimizer, epoch, init_lr):
    updated_lr = init_lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = updated_lr


def set_targets(data):
    for key0 in data:
        lo, hi = data[key0]['interval']

        targets = []
        for key1 in data:
            lo_tmp, hi_tmp = data[key1]['interval']
            target_candidate = (lo_tmp + hi_tmp) / 2.0
            if target_candidate > hi or target_candidate < lo:
                targets.append(target_candidate)
        
        data[key0]['targets'] = targets


def train(data, target, args):

    N = data['data_feat'].shape[0]
    print N

    X = torch.from_numpy(data['data_feat'].astype(np.float32))
    y = torch.ones(N,1) * target

    train = data_utils.TensorDataset(X, y)
    train_loader = data_utils.DataLoader(
        train, 
        batch_size=args.batch_size, 
        shuffle=True)

    model = AGAEncoderDecoder()
    model.set_phi(args.pretrained_phi, args.dim)
    model.set_rho(args.pretrained_rho, args.dim)
    if args.cuda:
        model.cuda()
    
    loss_fn_tmm = torch.nn.MSELoss()
    loss_fn_reg = torch.nn.MSELoss()
    if args.cuda:
        loss_fn_tmm.cuda()
        loss_fn_reg.cuda()

    optimizer = torch.optim.Adam(
        model.get_phi().parameters(), 
        lr=args.learning_rate)

    model.train()
    for epoch in range(1, args.epochs):
        epoch_loss = 0
    
        for i, (src, tgt) in enumerate(train_loader):
            
            if args.cuda:
                src = src.cuda()
                tgt = tgt.cuda()

            src_var = Variable(src)
            tgt_var = Variable(tgt)

            out_var = model(src_var)
            
            loss_tmm = loss_fn_tmm(out_var[1], tgt_var)
            loss_reg = loss_fn_reg(out_var[0], src_var)
            loss = 0.7*loss_reg + 0.3*loss_tmm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.data[0]

        if args.verbose:
            cprint('Loss [epoch=%.4d]=%.4f' % (epoch, epoch_loss), 'blue')

    return model.get_phi()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    with open(args.data_file) as fid:
        data = pickle.load(fid)

    set_targets(data)

    if not args.save is None:
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    phi_dict = {}
    for key in data:
        phi_dict[key] = {
            'interval': [],
            'targets':  [],
            'model_files': []
            }

        for target in data[key]['targets']:
            if args.verbose:
                cprint('Train [%.2f, %.2f] -> %.2f'  % (
                    data[key]['interval'][0],
                    data[key]['interval'][1],target))

            tmp_phi = train(
                data[key], 
                target, 
                args)
            
            out_model_file = str(uuid.uuid4()) + '.pht.tar'

            phi_dict[key]['targets'].append(target)
            phi_dict[key]['interval'].append(data[key]['interval'])
            phi_dict[key]['model_files'].append(out_model_file)

            if not args.save is None:
                torch.save(
                    tmp_phi.state_dict(), 
                    os.path.join(args.save, out_model_file))
            

if __name__ == "__main__":
    main()







    
