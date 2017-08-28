# AGA : Attribute-Guided Augmentation

This repository contains a PyTorch implementation of

```
@inproceedings{Dixit17a,
	author    = {M.~Dixit and R.~Kwitt and M.~Niethammer and N.~Vasconcelos},
	title     = {AGA : Attribute-Guided Augmentation},
	booktitle = {CVPR},
	year      = 2017}
```

- [Paper](http://www.svcl.ucsd.edu/publications/conference/2017/AGA/AGA_final.pdf)
- [Talk](https://youtu.be/F3ThW3RLSAU?t=26)
- [Presentation](https://drive.google.com/file/d/0BxHF82gaPzgSRVFXRWxHaTNjSlk/view?usp=sharing)

## Requirements

### Software

To reproduce our results on SUNRGBD from scratch, we need to following packages

- [Fast-RCNN](https://github.com/rbgirshick/fast-rcnn.git)
- [PyTorch](http://pytorch.org/)
- [LibLinear](https://github.com/ninjin/liblinear)
- [scikit-learn](http://scikit-learn.org/stable)

The *SelectiveSearch* code (downloadable) is included in this repository.
The download script is shamelessly taken from Shaoqing Ren's GitHub
[SPP_net](https://github.com/ShaoqingRen/SPP_net) repository. To use
*SelectiveSearch*, start MATLAB and run

```
cd <PATH_TO_AGA_DIR>/3rdparty/selective_search
fetch_selective_search
img = imread('<PATH_TO_AGA_DIR>/datasets/SUNRGBD/image_00001.jpg');
boxes = selective_search_boxes(img);
save('<PATH_TO_AGA_DIR>/datasets/SUNRGBD/image_00001_ss_boxes.mat', 'boxes');
```

### Data

We provide a couple of files that have already prepared SUNRGBD data and a
finetuned (to a selection of 20 object classes) Fast-RCNN model. These
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
that you have run *SelectiveSearch* (see paragraph above) on each image
(in MATLAB) and stored the bounding box proposals for each image as `image_00001_ss_boxes.mat` in the same folder (i.e., `/data/images`).

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

`object_features.py` uses the *SelectiveSearch* bounding boxes and
create (in `/data/images`), for each image, a file
`<IMAGE_NAME>_bbox_features.mat` which contains Fast-RCNN (default: FC7)
features with detection scores for each object class.

## Collecting data for AGA training

We create, for each image, a Python pickle file `<IMAGE_NAME>_fc7.pkl` that
contains Fast-RCNN FC7 activations for all bounding boxes that overlap with
the ground truth by IoU > 0.7 and detection scores for object classes > 0.5. Detections for `__background__` and `others` are excluded from this process.
Further, these files contain annotations (per remaining bounding box) for
*depth* and *pose*.

```bash
cd <PATH_TO_AGA_DIR>
python collect_train.py \
	--img_meta datasets/SUNRGBD/SUNRGBD_meta.pkl \
	--img_list /data/images/image_list.txt \
	--img_base /data/images \
	--bbox_postfix _ss_boxes.mat \
	--data_postfix _bbox_features.mat \
	--outfile_postfix _fc7.pkl \
	--object_class_file datasets/SUNRGBD/SUNRGBD_objects.pkl
```

## Training the object attribute strength predictor

To train the attribute strength predictor (here: for *depth*), we first
need to collect adequate trainign data:

```bash
mkdir /data/output
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
training data, collected from images 0 through 5335 (essentially, half of the data from
SUNRGBD). `collect_rho_data.py` also, by default, samples activations from all object
classes in a balanced manner (with sample size equal to the smallest number of activations per object class). We will later use the `--no_sampling` option to
create evaluation data for the attribute strength predictor.

Next, we can train the regressor (a simple MLP, see paper) (for 150 epochs):

```bash
cd <PATH_TO_AGA_DIR>
python train_rho.py \
	--log_file /data/output/rho_model_fc7.log \
	--data_file /data/output/rho_train_fc7.pkl \
	--save /data/output/rho_model_fc7.pht \
	--epochs 150 \
	--verbose
```

By default, we train with an initial learning rate of 0.001. The learning
schedule halves the learning rate every 50-th epoch.

## Training the synthesis function(s)

Now that we have a trained attribute strength predictor, we can go ahead
and train the synthesis function(s). For that, we again collect adequate
training data first. In detail, we collect activations from all objects with
attribute strengths within certain intervals. In case of *depth*, for example,
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
information about all trained models. The models are stored in the
corresponding folder `/data/output/phi_train_fc7`.

Before training the synthesis functions, we pretrain using all available
data that we previously collected for training the regressor. This step
simply trains the encoder-decoder to map from FC7 activations to FC7
activations. We found this to be beneficial, as training the synthesis
functions can be tricky if little data is available per interval.

```bash
cd <PATH_TO_AGA_DIR>
python pretrain_phi.py \
    --data_file /data/output/rho_train_fc7.pkl \
    --epochs 150 \
    --learning_rate 0.01 \
    --save /data/output/phi_model_fc7_pretrained.pht \
    --verbose
```

Then, we can start the *actual* training:

```bash
cd <PATH_TO_AGA_DIR>
python train_phi.py \
    --pretrained_rho /data/output/rho_model_fc7.pht \
    --pretrained_phi /data/output/phi_model_fc7_pretrained.pht \
    --data_file /data/output/phi_train_fc7.pkl \
    --save /data/output/phi_model_fc7 \
    --learning_rate 0.01  \
    --epochs 150 \
    --verbose
```

The trained models will be dumped to `/data/output/phi_model_fc7` and a
metadata file `/data/output/phi_model_fc7.pkl` will be created.

## Synthesis via AGA

To demonstrate synthesis, we reproduce the results from Table 3 of the [paper](http://www.svcl.ucsd.edu/publications/conference/2017/AGA/AGA_final.pdf).
For convenience, we provide a selection of images from SUNRGBD (left-out during training of the regressor + synthesis) that contain objects that we have never
seen before. In particular, the object classes are:
*picture, whiteboard, fridge, counter, books, stove, cabinet, printer, computer,
ottoman*. This is the *T0* set from the paper. The data can be downloaded from

- [Data for T0 object class set](https://drive.google.com/file/d/0BxHF82gaPzgSdEd0RTVmRXdyNDQ/view?usp=sharing)

We next synthesize FC7 activations (using our trained synthesis functions for attribute *depth*). To do so, we collect all data and then synthesize.

```bash
cd <PATH_TO_AGA_DIR>
python collect_eval.py \
    --img_list /data/T0/image_list.txt \
    --img_base /data/T0 \
    --data_postfix _bbox_features.mat \
    --label_postfix _ss_labels.mat \
    --object_class_file datasets/SUNRGBD/SUNRGBD_one_shot_classes_T0.pkl --outfile_postfix _fc7.pkl
```

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
This will create synthesis files `<IMAGE_NAME>_fc7_AGA_depth.pkl` for
ever image in `/data/T0` containing the synthesized features.

## Experiments

### One/Few-shot object recognition

Finally, we can run the one/few-shot object recognition experiment(s) from Table 3 of the paper.

```bash
cd <PATH_TO_AGA_DIR>/experiments
python SUNRGBD_few_shot.py \
	--img_list /data/T0/image_list.txt \
	--img_base /data/T0 \
	--data_postfix _fc7_AGA_depth.pkl \
	--runs 100
	--shots 1
```
This will run 100 trials of selecting one-shot instances from each object class
(in T0 we have 10 classes) and training a linear SVM and a 1-NN classifier using
(1) only one-shot instances as well as (2) one-shot instances *and* synthesized
features. Testing is done using all remaining original activations. The `--shots`
argument allows you to specify how many original samples you want to use for the
experiments. If, for example, you specify `--shots 5`, you can run a 5-shot 
object-recognition experiment. In that case, AGA-synthesized results will be based
on the original 5 samples per class + AGA-synthesized features.

*Note*: If you
specify the `--omit_original` flag, the same experiment is performed, but training
of the SVM and 1NN will only use synthetic data (without original samples).
adsf
This should produce (using the settings from above) **classification results**
(w/o AGA and w AGA) similar to:

```bash
[00001] SVM (w/o AGA): 35.76 | [00001] SVM (w AGA): 47.19 | [00001] 1NN (w/o AGA): 32.12 | [00001] 1NN (w AGA): 43.27 |
[00002] SVM (w/o AGA): 35.00 | [00002] SVM (w AGA): 40.06 | [00002] 1NN (w/o AGA): 35.14 | [00002] 1NN (w AGA): 40.48 |
[00003] SVM (w/o AGA): 45.73 | [00003] SVM (w AGA): 42.14 | [00003] 1NN (w/o AGA): 42.80 | [00003] 1NN (w AGA): 41.28 |
...
Average: SVM (w/o AGA): 35.96 | SVM (w AGA): 39.64 | 1NN (w/o AGA): 35.31 | 1NN (w AGA): 37.61 |
```

Lets run the same experiment **without** the one-shot instances included in the
synthetic data:

```bash
[00001] SVM (w/o AGA): 35.76 | [00001] SVM (w AGA): 43.55 | [00001] 1NN (w/o AGA): 32.12 | [00001] 1NN (w AGA): 41.57 |
[00002] SVM (w/o AGA): 41.57 | [00002] SVM (w AGA): 40.25 | [00002] 1NN (w/o AGA): 35.14 | [00002] 1NN (w AGA): 39.16 |
[00003] SVM (w/o AGA): 38.64 | [00003] SVM (w AGA): 41.99 | [00003] 1NN (w/o AGA): 42.80 | [00003] 1NN (w AGA): 40.01 |
...
Average: SVM (w/o AGA): 33.50 | SVM (w AGA): 40.12 | 1NN (w/o AGA): 35.31 | 1NN (w AGA): 36.73 |
```



## Differences to the paper

1. In the paper, we do not use an adaptive learning
rate strategy. Here, for all training, we half the learning rate every 50-th
epoch.

2. In addition to the results from the paper, we now also provide results for a
1-NN classifier as this *more directly* assesses the quality of the synthesis
results.
