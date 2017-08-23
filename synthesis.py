"""Implementation of the synthesis function (phi).

Author: rkwitt, mdixit (2017)
"""

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils

from misc.tools import build_file_list
from models import models

import os
import time
import pickle
import logging
from termcolor import colored, cprint
import argparse
import h5py
import numpy as np
import sys


def setup_parser():
    parser = argparse.ArgumentParser(description='Synthesis function')
    parser.add_argument(
        "--img_list",
        metavar='',
        help="list with image names (no extension)")
    parser.add_argument(
        "--img_base",
        metavar='',
        help="base directory for image data")
    parser.add_argument(
        "--data_postfix",
        metavar='',
        help="postfix + file extension to identity data files")
    parser.add_argument(
        "--syn_postfix",
        metavar='',
        help="postfix + file extension image file name")
    parser.add_argument(
        "--phi_model",
        metavar='',
        help="directory name for phi models")
    parser.add_argument(
        "--rho_model",
        metavar='',
        help="filename of trained rho model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="enables verbose output")
    parser.add_argument(
        "--dim",
        type=int,
        default=4096,
        help="dimensionality of features (default: 4096)")
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        default=False,
        help='disables CUDA training (default: False)')
    return parser


def get_phi_models(phi_dict, val):
    model_files = []
    for key in phi_dict:
        lo, hi = phi_dict[key]['interval']
        if val >= lo and val < hi:
            targets  = phi_dict[key]['targets']
            for i, target in enumerate(targets):
                model_files.append(
                    phi_dict[key]['model_files'][i])

    if len(model_files) == 0:
        cprint('No valid model file found {}'.format(val),'blue')
        return None
    else:
        return model_files


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    data_file_list = build_file_list(
        args.img_list,
        args.img_base,
        args.data_postfix)

    img_file_list = build_file_list(
        args.img_list,
        args.img_base)

    assert len(data_file_list) == len(img_file_list)

    with open(args.phi_model + '.pkl','r') as fid:
        phi_dict = pickle.load(fid)

    if args.verbose:
        cprint("Loaded phi info file {}".format(
                args.phi_model + '.pkl'),'blue')

    for cnt, data_file in enumerate(data_file_list):

        # load data file
        with open(data_file, 'r') as fid:
            data = pickle.load(fid)

        # check if we have valid features
        if not data['is_valid']:
            if args.verbose:
                cprint("Data file {} invalid ... skipping".format(
                    data_file), 'blue')
            continue

        # load attribute strength predictor + set to eval mode
        rho_model = models.rho(args.dim)
        rho_model.load_state_dict(torch.load(args.rho_model))
        if args.cuda:
            rho_model.cuda()
        rho_model.eval()

        # get all available features
        tmp = torch.from_numpy(
            data['CNN_activations'].astype(np.float32))

        #
        # step through one feature vector at a time and
        #   1) predict attribute
        #   2) select appropriate phi
        #   3) call phi for all available targets
        #
        syn_data = {}

        for i in np.arange(data['CNN_activations'].shape[0]):
            X = tmp[i,:].unsqueeze(0)
            if args.cuda:
                X = X.cuda()
            X = Variable(X)

            attr_pred = rho_model(X)
            attr_pred = attr_pred.data.cpu().numpy().flatten()

            model_files = get_phi_models(phi_dict,attr_pred[0])

            # if no appropriate phi is available, just keep the original features
            if model_files is None:
                if args.verbose:
                    cprint("No valid model for activation {} from {}".format(
                        i, data_file),'blue')
                syn_data[i] = {
                    'obj_idx': data['obj_idx'][i],
                    'CNN_activation_org': data['CNN_activations'][i,:].reshape(1,args.dim),
                    'CNN_activation_syn': None}
                continue

            CNN_activation_syn = np.zeros((len(model_files), args.dim))

            for j, model_file in enumerate(model_files):
                phi_model = models.phi(args.dim)
                phi_model.load_state_dict(
                    torch.load(os.path.join(args.phi_model, model_file)))
                if args.cuda:
                    phi_model.cuda()
                phi_model.eval()

                CNN_activation_syn[j,:] = phi_model(X).data.cpu().numpy()

            if args.verbose:
                cprint("Synthesized {} activations for {} of {}".format(
                    CNN_activation_syn.shape[0], i, data_file), 'blue')

            # store original + AGA-syn. features indexed by their occurrence
            syn_data[i] = {
                'obj_idx': data['obj_idx'][i],
                'CNN_activation_org': data['CNN_activations'][i,:].reshape(1,args.dim),
                'CNN_activation_syn': CNN_activation_syn}

        with open(img_file_list[cnt] + args.syn_postfix, 'w') as fid:
            pickle.dump(syn_data, fid, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
