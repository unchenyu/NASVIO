# NASVIO

This repository contains the training and evaluation code for *Search for efficient deep visual-inertial odometry through neural architecture search* and the searched models

## Data Preparation

The test dataset is KITTI Odometry dataset. The IMU data after pre-processing is provided under `data/imus`. To download the images and poses, please run

      $cd data
      $source data_prep.sh 

## Pretrained checkpoints on searched best models

Two checkpoints with low FLOPS target (`flops_target.zip`) and low latency target (`latency_target.zip`) are provided. Simply unzip to retrieve the checkpoints.

## Test the pretrained models

Select which model to run by changing the `self.target` parameter (`flops` or `latency`) in the `params.py`. Then run:

      python3 test.py 
