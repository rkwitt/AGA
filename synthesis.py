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
        "--img_list",           
        metavar='', 
        help="image list to process")
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

if __name__ == "__main__":
    main()







    
