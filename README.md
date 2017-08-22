# AGA : Attribute-Guided Augmentation

This repository contains a PyTorch implementation of

```
@inproceedings{Dixit17a,	
	author    = {M.~Dixit and R.~Kwitt and M.~Niethammer and N.~Vasconcelos},
	title     = {AGA : Attribute-Guided Augmentation},
	booktitle = {CVPR},
	year      = 2017}
```

**Disclaimer**: This documentation is still in development! 

## Requirements

### Software

To reproduce our results on SUNRGBD from scratch, we need to following packages

- [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn.git)
- [SelectiveSearch](https://github.com/sergeyk/selective_search_ijcv_with_python0)
- [PyTorch](http://pytorch.org/)
- [LibLinear](https://github.com/ninjin/liblinear)
- [scikit-learn](http://scikit-learn.org/stable)

### Data

We provide a couple of files that have already prepared SUNRGBD data and a
Fast-RCNN model, finetuned to a selection of SUNRGBD object classes. These
files can we downloaded via

- [SUNRGBD images](https://drive.google.com/file/d/0BxHF82gaPzgScVJ5LXVCWkRsaE0/view?usp=sharing)
- [Finetuned VGG16 model](https://drive.google.com/file/d/0BxHF82gaPzgSY2gySFpQU0hJOVU/view?usp=sharing)
- [Original SUNRGBD metadata (MATLAB)](https://drive.google.com/file/d/0BxHF82gaPzgSWFdPeEdSYnkxanM/view?usp=sharing)

For the original SUNRGBD metadata, see also [here](http://rgbd.cs.princeton.edu/).

We will assume, from now own, that the images are unpacked into `/data/images/`. The 
finetuned model has to be put into the `<PATH_TO_FAST_RCNN_DIR>/data/fast_rcnn_models`
folder.

## Running Fast-RCNN

Say, you have unpacked the original SUNRGBD images to `/data/images`, i.e., the 
directory should contain `image_00001.jpg` to `image_10335.jpg`. We also assume
that you have run *SelectiveSearch* on each image (in MATLAB) and stored the 
bounding box proposals for each image as `image_00001_ss_boxes.mat` in the same
folder (i.e., `/data/images`).

First, we create a file which contains all image filenames without extension.

```bash
cd /data/images
find . -name '*.jpg' -exec basename {} .jpg \; > image_list.txt
```

Next, we run the Fast-RCNN detector with

```bash
cd <PATH_TO_AGA_DIR>
cp misc/object_features.py <PATH_TO_FAST_RCNN_DIR>/tools
cd <PATH_TO_FAST_RCNN_DIR>
python tools/object_features.py
```

This generates (in `/data/images`), for each image, a file `<IMAGE_NAME>_bbox_features.mat` file
which contains Fast-RCNN (FC7) features with detection scores for each object class that we 
finetuned on.

## Collecting data for AGA training

```bash
cd <PATH_TO_AGA_DIR>
python collect_train.py \ 
	--img_meta datasets/SUNRGBD/SUNRGBD_meta.pkl \
	--img_list /data/images/allimgs.txt \
	--img_base /data/images \
	--bbox_postfix _ss_boxes.mat \
	--data_postfix _bbox_features.mat \
	--outfile_postfix _fc7.pkl \
	--object_class_file datasets/SUNRGBD/SUNRGBD_objects.pkl
```

This creates, for each image, a Python pickle file `<IMAGE_NAME>_fc7.pkl` that contains
Fast-RCNN FC7 activations for all bounding boxes that overlap with the ground truth by
IoU > 0.7 and detection scores for object classes > 0.5. Detections for `__background__`
and `others` are also excluded. Further, these files contain annotations (per remaining
bounding box) for *depth* and *pose*.

## Training object attribute strength predictor

To train the attribute strength predictor (here: for *depth*), we first need to collect
adequate data:

```bash
cd <PATH_TO_AGA_DIR>
 python collect_rho_data.py \
 	--img_list /data/images/image_list.txt \
	--img_base /data/images \
	--beg_index 0 \
	--end_index 5335 \
	--attribute depth \
	--data_postfix _fc7.pkl \
	--save /data/output/rho_train_fc7.pkl \
	--verbose 
```

This will create a file `/data/output/rho_train_fc7.pkl` which contains all relevant 
training data, collected from images 0 through 5335 (essentially half of the data from
SUNRGBD). `collect_rho_data.py` also, by default, samples activations from all object
classes in a balanced manner (with sample size equal to the smallest number of activations
per object class). We will later use the `--no_sampling` option to create evaluation 
data for the attribute strength predictor.

Next, we can run the training (for 150 epochs):

```bash
cd <PATH_TO_AGA_DIR>
python train_rho.py \ 
	--log_file /data/output/rho_model_fc7.log \ 
	--data_file /data/output/rho_train_fc7.pkl \ 
	--save /data/output/rho_model_fc7.pht \
	--epochs 150 \
	--verbose
```

By default, this trains with a learning rate of 0.001. The learning 
schedule halves the learning rate every 50-th epoch.

## Training the synthesis function

Now that we have a trained attribute strength predictor, we can go ahead
and train the synthesis function(s). For that, we first collect adequate
training data. In detail, we collect activations from all objects with 
attribute strenght within certain intervals. In case of depth, for example,
these intervals are [0m,1m], [0.5m,1.5m], ..., [4.5m,5.5m], but other 
binning strategies are of course possible. Our predefined interval file is 
`<PATH_TO_AGA_DIR>/datasets/SUNRGBD/SUNRGBD_depth_intervals.txt`. We also
ensure to only use data from objects that have at least 100 activations 
in an interval.

```bash
cd <PATH_TO_AGA_DIR>
python collect_phi_data.py \
	--interval_file datasets/SUNRGBD/SUNRGBD_depth_intervals.txt \
	--img_list /data/images/image_list.txt \
	--img_base /data/images \
	--beg_index 0 \
	--end_index 5335 \
	--data_postfix _fc7.pkl \
	--attribute depth \
	--save /data/output/phi_train_fc7.pkl \
	--min_count 100 \
	--verbose 
```

This will create a file `/data/output/phi_train_fc7.pkl` that will hold 
information about all trained models (stored in the folder `/data/output/phi_train_fc7`).

## Synthesis via AGA

To demonstrate synthesis, we reproduce the results from Table 3 of the [paper](http://www.svcl.ucsd.edu/publications/conference/2017/AGA/AGA_final.pdf). For 
this, we provide a selection of images from SUNRGBD (left-out during training)
that contain objects that we have never seen before. In particular, 
*picture, whiteboard, fridge, counter, books, stove, cabinet, printer, computer, 
ottoman*. The data can be downloaded from:

- [Data for T0 object class set](https://drive.google.com/file/d/0BxHF82gaPzgSdEd0RTVmRXdyNDQ/view?usp=sharing)

We next, synthesize FC7 activations (using our synthesis functions for attribute *depth*):

```bash
cd <PATH_TO_AGA_DIR>
python synthesis.py \
	--img_list /data/T0/image_list.txt \
	--img_base /data/T0 \
	--phi_model /data/output/phi_model_fc7 \
	--rho_model /data/output/rho_model_fc7.pht \
	--data_postfix _fc7.pkl \
	--syn_postfix _fc7_AGA_depth.pkl
	--verbose 
```
This will create additional files `<IMAGE_NAME>_fc7_AGA_depth.pkl` for ever image in `/data/T0`
that contain the synthesized features.

## Experiments

### One-shot object recognition

Finally, we can run the one-shot object recognition experiment from Table 3 of the paper. 

```bash
cd <PATH_TO_AGA_DIR>/experiments
python SUNRGBD_one_shot.py \ 
	--img_list /data/T0/image_list.txt \
	--img_base /data/T0 \
	--data_postfix _fc7_AGA_depth.pkl \
	--runs 100
```
This will run 100 trials of selecting one-shot instances from each object class
(in T0 we have 10 classes) and training a linear SVM and a 1-NN classifier using
(1) only one-shot instances as well as (2) one-shot instances *and* synthesized 
features. Testing is done using all remaining original activations. *Note*: If you 
specify the `--omit_original` flag, the same experiment is performed, but training 
of the SVM and 1NN will only use synthetic data (without original samples).

















