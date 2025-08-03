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
from deepvio.helper import normalize_angle_delta


def get_data_info(folder_list, seq_len, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True, is_right=False):
    X_path, Y, I, I_int = [], [], [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()

        # Load & sort the raw data
        poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
        imus = sio.loadmat('{}{}.mat'.format(par.imu_dir, folder))['imu_data_interp']
        # imus = sio.loadmat('{}{}/imu.mat'.format(par.image_dir, folder))['imu_data_interp']
        fpaths = glob.glob('{}{}/image_2/*.png'.format(par.image_dir, folder))        
        fpaths.sort()
        
        x_segs, y_segs, i_segs, i_int_segs = [], [], [], []

        # Fixed seq_len   
        n_frames = len(fpaths)
        jump = seq_len - overlap
        start = 0
        while start + seq_len <= n_frames:
            x_segs.append(fpaths[start:start+seq_len])
            y_segs.append(poses[start:start+seq_len-1])
            
            # Dividing IMU raw data
            start_tmp = start*par.imu_per_image-par.imu_prev-1
            stop_tmp = (start+seq_len-1)*par.imu_per_image+2                    
            if start_tmp < 0:  
                padded = np.concatenate((np.zeros((-start_tmp, 6)), imus[:stop_tmp]))
                i_segs.append(padded.tolist())
            elif stop_tmp > imus.shape[0]:  # If stopping point after the end of imu sequences
                padded = np.concatenate((imus[start_tmp:], np.zeros((stop_tmp-imus.shape[0], 6))))
                i_segs.append(padded.tolist())
            else:
                i_segs.append(imus[start_tmp:stop_tmp])

            start += jump
        
        if is_right:
            fpaths = glob.glob('{}{}/image_3/*.png'.format(par.image_dir, folder))
            fpaths.sort()
            # Fixed seq_len   
            n_frames = len(fpaths)
            jump = seq_len - overlap
            start = 0
            while start + seq_len <= n_frames:
                x_segs.append(fpaths[start:start+seq_len])
                y_segs.append(poses[start:start+seq_len-1])
                
                # Dividing IMU raw data
                start_tmp = start*par.imu_per_image-par.imu_prev-1
                stop_tmp = (start+seq_len-1)*par.imu_per_image+2                    
                if start_tmp < 0:  
                    padded = np.concatenate((np.zeros((-start_tmp, 6)), imus[:stop_tmp]))
                    i_segs.append(padded.tolist())
                elif stop_tmp > imus.shape[0]:  # If stopping point after the end of imu sequences
                    padded = np.concatenate((imus[start_tmp:], np.zeros((stop_tmp-imus.shape[0], 6))))
                    i_segs.append(padded.tolist())
                else:
                    i_segs.append(imus[start_tmp:stop_tmp])

                start += jump

        Y += y_segs
        X_path += x_segs
        I += i_segs
        X_len += [len(xs) for xs in x_segs]

        print('Folder {} finish in {} sec'.format(folder, time.time() - start_t))

    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y, 'imu': I}
    df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'pose', 'imu'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe):
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)
        self.imu_arr = np.asarray(self.data_info.imu)
        self.imu_len = (par.seq_len[0]-1)*10 + 3 + par.imu_prev
    
    def __getitem__(self, index):
        
        flag_hflip = random.random() < 0.5 if par.is_hflip else False
        flag_color = random.random() < 0.5 if par.is_color else False
        
        # If apply the color augmentation
        if flag_color:
            bright_fact = torch.rand(1)*0.4+0.8        # Brightness factor (0.8, 1.2)
            contrast_fact = torch.rand(1)*0.4+0.8        # Contrast factor (0.8, 1.2)
            saturation_fact = torch.rand(1)*0.4+0.8      # Saturation factor (0.8, 1.2)
            hue_fact = torch.rand(1)*0.2-0.1           # Hue factor (-0.1, 0.1)
        
        if par.flag_imu_aug:
            shift = random.uniform(-0.1, 0.1)
        
        if par.is_crop:
            x_scaling, y_scaling = 1.1, 1.1
            scaled_h, scaled_w = int(par.img_h * y_scaling), int(par.img_w * x_scaling)
            offset_y = np.random.randint(scaled_h - par.img_h + 1)
            offset_x = np.random.randint(scaled_w - par.img_w + 1)

        # Prepare to load the images
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        
        image_sequence = []
        image_sequence_LR = []
        for idx, img_path in enumerate(image_path_sequence):

            img_as_img = Image.open(img_path)
            
            # Cropping
            if par.is_crop:
                img_as_img = TF.resize(img_as_img, size=(scaled_h, scaled_w))
                img_as_img = TF.crop(img_as_img, top=offset_y, left=offset_x, height=par.img_h, width=par.img_w)
            else:
                img_as_img = TF.resize(img_as_img, size=(par.img_h, par.img_w))

            # Colar augmentation
            if flag_color:
                img_as_img = TF.adjust_brightness(img_as_img, bright_fact)
                img_as_img = TF.adjust_contrast(img_as_img, contrast_fact)
                img_as_img = TF.adjust_saturation(img_as_img, saturation_fact)
                img_as_img = TF.adjust_hue(img_as_img, hue_fact)

            # Horizontal flips
            if flag_hflip:
                img_as_img = TF.hflip(img_as_img)
            
            img_as_img_lr = TF.resize(img_as_img, size=(par.img_h//4, par.img_w//4))
            
            # Convert to tensor
            img_as_tensor_lr = TF.to_tensor(img_as_img_lr)-0.5
            img_as_tensor_lr = img_as_tensor_lr.unsqueeze(0)
            img_as_tensor = TF.to_tensor(img_as_img)-0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
            image_sequence_LR.append(img_as_tensor_lr)

        image_sequence = torch.cat(image_sequence, 0)
        image_sequence_LR = torch.cat(image_sequence_LR, 0)

        if par.flag_imu_aug:
            fp = self.imu_arr[index]
            xp = np.arange(self.imu_len)
            f = interpolate.interp1d(xp, fp, axis=0)
            x_new = np.arange(1, self.imu_len-1) + shift
            imu_seq = f(x_new)
        else:
            imu_seq = self.imu_arr[index][1:-1]

        imu_sequence = torch.FloatTensor(imu_seq)
        
        # Flip the angular velocity and accelaration
        if flag_hflip:
            imu_sequence[:, 1] = -imu_sequence[:, 1]
            imu_sequence[:, 5] = -imu_sequence[:, 5]
            imu_sequence[:, 3] = -imu_sequence[:, 3]
        
        # Prepare the ground truth pose
        gt_sequence = self.groundtruth_arr[index][:, :6]
        gt_sequence = torch.FloatTensor(gt_sequence)
        
        # Normalize all angles
        for gt_seq in gt_sequence:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])
            gt_seq[1] = normalize_angle_delta(gt_seq[1])
            gt_seq[2] = normalize_angle_delta(gt_seq[2])

        # If apply horizontal flip
        if flag_hflip:
            for gt_seq in gt_sequence:
                gt_seq[1], gt_seq[2], gt_seq[3] = -gt_seq[1], -gt_seq[2], -gt_seq[3]  # Flip the Yaw angle
                
        return (sequence_len, image_sequence, image_sequence_LR, imu_sequence, gt_sequence)

    def __len__(self):
        return len(self.data_info.index)


class ImageSequenceDataset_test(Dataset):
    def __init__(self, info_dataframe):
        # Transforms
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)
        self.imu_arr = np.asarray(self.data_info.imu)
        
    def __getitem__(self, index): 
        
        gt_sequence = self.groundtruth_arr[index][:, :6]
        gt_sequence = torch.FloatTensor(gt_sequence)

        for gt_seq in gt_sequence:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])
            gt_seq[1] = normalize_angle_delta(gt_seq[1])
            gt_seq[2] = normalize_angle_delta(gt_seq[2])
          
        # Prepare to load the images
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  # sequence_len = torch.tensor(len(image_path_sequence))
        
        if par.is_crop:
            x_scaling, y_scaling = 1.1, 1.1
            scaled_h, scaled_w = int(par.img_h * y_scaling), int(par.img_w * x_scaling)

        image_sequence = []
        image_sequence_LR = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            
            if par.is_crop:
                img_as_img = TF.resize(img_as_img, size=(scaled_h, scaled_w))
                img_as_img = TF.center_crop(img_as_img, (par.img_h, par.img_w))
            else:
                img_as_img = TF.resize(img_as_img, size=(par.img_h, par.img_w))

            img_as_img_lr = TF.resize(img_as_img, size=(par.img_h//4, par.img_w//4))
            img_as_tensor = TF.to_tensor(img_as_img)-0.5
            img_as_tensor_lr = TF.to_tensor(img_as_img_lr)-0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            img_as_tensor_lr = img_as_tensor_lr.unsqueeze(0)

            image_sequence.append(img_as_tensor)
            image_sequence_LR.append(img_as_tensor_lr)

        image_sequence = torch.cat(image_sequence, 0)
        image_sequence_LR = torch.cat(image_sequence_LR, 0)

        # IMU data
        imu_sequence = torch.FloatTensor(self.imu_arr[index])[1:-1, :]
        return (sequence_len, image_sequence, image_sequence_LR, imu_sequence, gt_sequence)

    def __len__(self):
        return len(self.data_info.index)


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


# Example of usage
if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    sample_times = 1
    folder_list = ['00']
    seq_len_range = [11, 11]
    df = get_data_info(folder_list, seq_len_range[0], overlap, sample_times)
    print('Elapsed Time (get_data_info): {} sec'.format(time.time() - start_t))
    # Customized Dataset, Sampler
    n_workers = 4
    dataset = ImageSequenceDataset(df)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    print('Elapsed Time (dataloader): {} sec'.format(time.time() - start_t))

    for batch in dataloader:
        s, x, y, _, _ = batch
        print('=' * 50)
        #print('len:{}\nx:{}\ny:{}\nz:{}'.format(s, x.shape, y.shape, z.shape))
        import pdb; pdb.set_trace()  # breakpoint 9dd4e86a //


    print('Elapsed Time: {} sec'.format(time.time() - start_t))
    print('Number of workers = ', n_workers)