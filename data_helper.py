import os
import time
import glob
import pandas as pd
import numpy as np
from PIL import Image
import random
import scipy.io as sio
from scipy import interpolate

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import torchvision.transforms.functional as TF

from params import par


def get_data_info_test(folder, seq_len):
    X_path, Y, I = [], [], []

    start_t = time.time()

    # Load & sort the raw data
    poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
    fpaths = glob.glob('{}{}/image_2/*.png'.format(par.image_dir, folder))
    imus = sio.loadmat('{}{}.mat'.format(par.imu_dir, folder))['imu_data_interp']
    # imus = sio.loadmat('{}{}/imu.mat'.format(par.image_dir, folder))['imu_data_interp']   
    fpaths.sort()
    
    n_frames = len(fpaths)
    start = 0
    while start + seq_len < n_frames:
        x_seg = fpaths[start:start+seq_len]
        X_path.append(x_seg)
        Y.append(poses[start:start+seq_len-1])

        start_tmp = start*par.imu_per_image-par.imu_prev
        stop_tmp = (start+seq_len-1)*par.imu_per_image+1
        if start_tmp < 0:  # If starting point before the start of imu sequences
            pad_zero = np.zeros((-start_tmp, 6))
            padded = np.concatenate((pad_zero, imus[:stop_tmp]))
            I.append(padded.tolist())
        else:
            I.append(imus[start_tmp:stop_tmp])
        start += seq_len - 1
    
    X_path.append(fpaths[start:])
    Y.append(poses[start:])
    I.append(imus[start*par.imu_per_image-par.imu_prev:])
    
    print('Folder {} finish in {} sec'.format(folder, time.time() - start_t))
    # Convert to pandas dataframes
    data = {'image_path': X_path, 'pose': Y, 'imu': I}
    df = pd.DataFrame(data, columns=['image_path', 'pose', 'imu'])
    return df
