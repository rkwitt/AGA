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
    parser = argparse.ArgumentParser(description="Pretrain the encoder-decoder.")
    parser.add_argument(
        "--data_file",        
        metavar='', 
        help='data input file (pickle)')
    parser.add_argument(
        "--save",        
        metavar='', 
        help='filename of saved model')
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
        default=300, 
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


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model = models.phi_model(args.dim)

    if args.verbose: 
        print model
    if args.cuda:
        model.cuda()

    with open(args.data_file) as fid:
        data = pickle.load(fid)

    if args.verbose:
        cprint('Using CUDA: %s' % bool(args.cuda), 'blue')
        cprint('Data: %d x %d'  % data["data_feat"].shape)

    X = torch.from_numpy(data['data_feat'].astype(np.float32))

    train = data_utils.TensorDataset(X, X)
    train_loader = data_utils.DataLoader(
        train, 
        batch_size=args.batch_size, 
        shuffle=True)

    loss_fn = torch.nn.MSELoss(size_average=True)
    if args.cuda:
        loss_fn.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate)
    
    model.train()
    for epoch in range(1, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate)
        epoch_loss = 0
    
        for i, (src, tgt) in enumerate(train_loader):
            
            if args.cuda:
                src = src.cuda()
                tgt = tgt.cuda()
            
            src_var = Variable(src)
            tgt_var = Variable(tgt)

            out_var = model(src_var)
            loss = loss_fn(out_var, tgt_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.data[0]

        cprint('Loss [epoch=%.4d]=%.4f' % (epoch, epoch_loss), 'blue')

    if not args.save is None:
        torch.save(model.state_dict(), args.save)



if __name__ == "__main__":
    main()







    
