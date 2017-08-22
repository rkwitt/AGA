"""
Collect valid detections and augment by attribute data.

Author: rkwitt, mdixit (2017)
"""

from misc.AGAImage import AGAImage
from misc.tools import IoU, build_file_list, compute_SUNRGBD_attributes

from termcolor import colored, cprint
from scipy.io import loadmat
import argparse
import glob
import pickle
import numpy as np
import time
import sys
import os


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = argparse.ArgumentParser(description="Collect valid detection/activation data.")

    parser.add_argument(
        "--img_meta",      
        metavar='', 
        help="pickle file with image meta information")
    parser.add_argument(
        "--img_list",      
        metavar='', 
        help="file with image filenames (no extension)")
    parser.add_argument(
        "--img_base",      
        metavar='', 
        help="base directory of image files")
    parser.add_argument(
        "--ovl_thr",       
        metavar='', 
        type=float, 
        default=0.5, 
        help="overlap threshold when to keep detection BBs")
    parser.add_argument(
        "--det_thr",       
        metavar='', 
        type=float, 
        default=0.5, 
        help="detection threshold when to keep detection BBs")
    parser.add_argument(
        "--bbox_postfix",  
        metavar='', 
        help="postfix to identify bounding box files")
    parser.add_argument(
        "--object_class_file",  
        metavar='', 
        help='pickeled file with a list of object names')
    parser.add_argument(
        "--data_postfix",  
        metavar='', 
        help="postfix to identify feature + score files")
    parser.add_argument(
        "--outfile_postfix",  
        metavar='', 
        help="postfix to append for generated output files")
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=False, 
        dest="verbose", 
        help="Verbose output")
    return parser


def collect_data(img_meta, img_bbox_file, img_data_file, object_classes, ovl_thr=0.5, det_thr=0.5, exclude = ['__background__', '__others__']):
    detections = {
        'is_valid'       : False,   # is detection entry valid
        'CNN_scores'     : None,    # detection scores / object class
        'CNN_activations': None,    # CNN activations / bounding box
        'attributes'     : None,    # attributes associated with bounding box
        'det_idx'        : None,    # index of kept detection box
        'det_box'        : None,    # coordinates of detection box
        'gtb_idx'        : None,    # index of most-overlapping ground truth box
        'gtb_box'        : None,    # coordinates of most-overlapping ground truth box 
        'obj_cls'        : None,    # object classes
        'obj_idx'        : None,    # indices of object classes
        }

    bb1 = loadmat(img_bbox_file)['boxes']   # detected 2D bounding boxes
    bb2 = img_meta.get_gtBB2D()             # ground truth 2D bounding boxes
    tmp = loadmat(img_data_file)            # features + scores for all detections

    if bb2 is None or bb1 is None:        
        return detections

    Npr = bb1.shape[0] # nr. of 2D proposal bounding boxes
    Ngt = bb2.shape[0] # nr. of 2D ground truth bounding boxes

    # compute IoU scores for all detections vs. ground truth bounding boxes
    IoU_scores = np.zeros((Npr,Ngt))
    for i in np.arange(Npr):
        for j in np.arange(Ngt):
            x1 = bb2[j,0]
            y1 = bb2[j,1]
            x2 = bb2[j,2]
            y2 = bb2[j,3]
            IoU_scores[i,j] = IoU(bb1[i,:], np.array([x1,y1,x1+x2,y1+y2]))
    
    max_ovl = np.amax(IoU_scores, axis=1)       # max. overlap per detected bounding box
    row_max = np.where(max_ovl >= ovl_thr)[0]   # max. overlap > threshold

    det_idx = None
    gtb_idx = None
    CNN_sco = None
    CNN_act = None
    det_box = None
    obj_cls = None
    obj_idx = None
    attributes = []

    for cnt, i in enumerate(row_max): 
        idx = np.argmax(IoU_scores[i,:])        
        # idx holds the ground truth bbox index that overlaps
        # most with the current proposal (which already overlaps 
        # more than 0.5
        
        tmp_sco = tmp['CNN_scores'][i,:]
        tmp_act = tmp['CNN_feature'][i,:]
        
        #print cnt, i, compute_SUNRGBD_attributes(img_meta, idx)

        # make sure detections below a certain detection threshold as well
        # as dections corresponding to the exclude object classes are not
        # included.
        top = np.argmax(tmp_sco)
        if tmp_sco[top] < det_thr: continue
        if any(object_classes[top] in s for s in exclude): continue
            
        if not detections['is_valid']:
            obj_idx = [top]
            obj_cls = [object_classes[top]]
            det_idx = [i]
            gtb_idx = [idx]
            CNN_act = np.asmatrix(tmp_act)
            CNN_sco = np.asmatrix(tmp_sco)
            det_box = np.asmatrix(bb1[i,:])
            detections['is_valid'] = True
        else:
            CNN_sco = np.vstack((CNN_sco, tmp_sco))
            CNN_act = np.vstack((CNN_act, tmp_act))
            det_box = np.vstack((det_box, bb1[i,:]))
            obj_cls = obj_cls + [object_classes[top]]
            obj_idx = obj_idx + [top]
            det_idx = det_idx + [i]
            gtb_idx = gtb_idx + [idx]

        attributes.append(compute_SUNRGBD_attributes(img_meta, idx))

    if not detections['is_valid']:
        return detections

    # at this point, we only have valid detections
    detections['CNN_scores'] = CNN_sco
    detections['CNN_activations'] = CNN_act
    detections['attributes'] = attributes
    detections['det_idx'] = det_idx
    detections['gtb_idx'] = gtb_idx
    detections['gtb_box'] = bb2
    detections['det_box'] = det_box
    detections['obj_idx'] = obj_idx
    detections['obj_cls'] = obj_cls
    
    return detections

    
def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = setup_parser().parse_args()

    # load meta data for image database
    with open(args.img_meta, 'rb') as input:
        img_meta = pickle.load(input)

    if args.verbose:
        cprint('Loaded %s' % args.img_meta, 'blue')
    
    # load names of object classes
    with open(args.object_class_file, 'r') as input: 
        object_class_names = pickle.load(input)

    # compose filenames of images/bounding box proposals/feature files
    img_files = build_file_list(args.img_list)
    img_bbox_files = build_file_list(args.img_list, base=args.img_base, postfix=args.bbox_postfix)
    img_feat_files = build_file_list(args.img_list, base=args.img_base, postfix=args.data_postfix)

    for cnt,_ in enumerate(img_files):

        # compose output filename
        outfile = os.path.join(
            args.img_base, 
            img_files[cnt] + args.outfile_postfix)
        
        #if os.path.exists(outfile):
        #   if args.verbose:
        #       cprint("%s exists ... skipping" % outfile, 'blue')
        #   continue   
    
        t0 = time.time()
        data = collect_data(
            img_meta[cnt], 
            img_bbox_files[cnt], 
            img_feat_files[cnt], 
            object_class_names,
            ovl_thr=args.ovl_thr,
            det_thr=args.det_thr,
            exclude = ['__background__', '__others__'])
        ela = time.time() - t0

        if args.verbose:
          if data['is_valid']:
                cprint('Done with %s [%d, in %.2f s]' % (
                  img_files[cnt], 
                  data['CNN_activations'].shape[0], 
                  ela), 'blue')
        
        with open(outfile, 'wb') as fid:
            pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


