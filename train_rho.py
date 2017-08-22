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
    parser = argparse.ArgumentParser(description='Training of attribute strength regressor')
    parser.add_argument(
        "--log_file",
        metavar='',
        help="log file to use")
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
        "--save",
        metavar='',
        help="filename of output file")
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="enables verbose output")
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

    if args.verbose:
        cprint('Using CUDA: %s' % bool(args.cuda), 'blue')

    args.use_log = False
    if not args.log_file is None:
        logging.basicConfig(filename=args.log_file,level=logging.DEBUG)
        args.use_log = True

    with open(args.data_file) as fid:
        data = pickle.load(fid)

    data_attr = data['data_attr'].astype(np.float32).reshape((data['data_attr'].shape[0],1))
    data_feat = data['data_feat'].astype(np.float32)

    N_data_feat, D_data_feat = data_feat.shape
    N_data_attr, D_data_attr = data_attr.shape

    assert N_data_feat == N_data_attr

    if args.verbose:
        cprint('Data: (%d x %d)' % (N_data_feat, D_data_feat), 'blue')
        cprint('Attribute target: (%d x %d)' % (N_data_attr, D_data_attr), 'blue' )

    data_feat_th = torch.from_numpy(data_feat)
    data_attr_th = torch.from_numpy(data_attr)

    train = data_utils.TensorDataset(data_feat_th, data_attr_th)
    train_loader = data_utils.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True)

    model = models.rho(args.dim)
    if args.verbose:
        print model
    if args.cuda:
        model.cuda()

    loss_fn = torch.nn.MSELoss(size_average=False)
    if args.cuda:
        loss_fn.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)

    model.train()
    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch, args.learning_rate)
        epoch_loss = 0

        for i, (src, tgt) in enumerate(train_loader):

            if args.cuda:
                src = src.cuda()
                tgt = tgt.cuda()

            src_var = Variable(src)
            tgt_var = Variable(tgt)

            optimizer.zero_grad()

            out_var = model(src_var)
            loss = loss_fn(out_var, tgt_var)
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.data[0]

        if args.use_log:
            logging.info('%.4f' % epoch_loss)

        if args.verbose:
            cprint('Loss [epoch=%.4d]=%.4f' % (epoch, epoch_loss/data_feat.shape[0]), 'blue')

    model.eval()
    all = Variable(data_feat_th.cuda())
    out = model(all).cpu()
    
    #Z = np.zeros((out.data.numpy().shape[0],2))
    #Z[:,0] = out.data.numpy().ravel()
    #Z[:,1] = data_attr_th.numpy().ravel()
    #np.savetxt('/tmp/rho_pred.txt', Z, delimiter=' ')
    #print out.data.numpy(), data_attr_th.numpy()
    
    mae = np.abs(out.data.numpy() - data_attr_th.numpy()).mean()
    if args.verbose:
        cprint('MAE [train]=%.2f [m]' % mae, 'blue')

    if not args.save is None:
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()
