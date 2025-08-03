import os
import glob
import numpy as np
import time
import math
import scipy.io as sio

import cv2
from PIL import Image
import torch
from torchvision import transforms

from params import par
from deepvio.helper import R_to_angle, cal_rel_pose, angle_to_R
from deepvio.helper import so3_to_SO3, SO3_to_so3, euler_from_matrix_new, eulerAnglesToRotationMatrix

# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data():
    info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
    start_t = time.time()
    for video in info.keys():
        fn = '{}{}.txt'.format(par.pose_dir, video)
        print('Transforming {}...'.format(fn))
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = []
            for i in range(len(lines)):
                values = [float(value) for value in lines[i].split(' ')]
                if i > 0:
                    values_pre = [float(value) for value in lines[i-1].split(' ')]
                    poses.append(cal_rel_pose(values_pre, values)) 
            poses = np.array(poses)
            import pdb; pdb.set_trace()  # breakpoint c4b7e14e //
            
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn+'.npy', poses)
            print('Video {}: shape={}'.format(video, poses.shape))
    print('elapsed time = {}'.format(time.time()-start_t))

# Calculate the preintegration vector for IMU data
def create_pre_integration():
    info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
    start_t = time.time()
    delta = 0.01
    for video in info.keys():
        fn = '{}{}/imu.mat'.format(par.image_dir, video)
        imus = sio.loadmat(fn)['imu_data_interp']
        print('Transforming {}...'.format(fn))

        pre_ints = []
        for i in range(info[video][1]-info[video][0]):
            R_acc = np.eye(3)
            v_acc = np.zeros(3)
            p_acc = np.zeros(3)
            R_acc_inv = np.eye(3)
            v_acc_inv = np.zeros(3)
            p_acc_inv = np.zeros(3)
            for j in range(10):
                w = imus[(i-1)*10+j][3:]
                a = imus[(i-1)*10+j][:3]
                p_acc += v_acc*delta+0.5*R_acc.dot((delta**2)*a)
                v_acc += R_acc.dot(delta*a)
                delta_R = so3_to_SO3(delta*w)
                R_acc = R_acc@delta_R
            r_acc = SO3_to_so3(R_acc)
            pre_int = np.concatenate((r_acc, v_acc, p_acc))
            assert pre_int.sum() is not None and pre_int.sum() != np.inf and pre_int.sum() != -np.inf
            pre_ints.append(pre_int)
        pre_ints = np.stack(pre_ints)
        base_fn = os.path.splitext(fn)[0]
        np.save(base_fn+'_preint.npy', pre_ints)
        print('Video {}: shape={}'.format(video, pre_ints.shape))
    print('elapsed time = {}'.format(time.time()-start_t))


def calculate_rgb_mean_std(image_path_list, minus_point_5=False):
    n_images = len(image_path_list)
    cnt_pixels = 0
    print('Numbers of frames in training dataset: {}'.format(n_images))
    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    to_tensor = transforms.ToTensor()

    image_sequence = []
    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        cnt_pixels += img_as_np.shape[1]*img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
    mean_tensor =  [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    print('mean_tensor = ', mean_tensor)
    print('mean_np = ', mean_np)

    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]
    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        for c in range(3):
            tmp = (img_as_tensor[c] - mean_tensor[c])**2
            std_tensor[c] += float(torch.sum(tmp))
            tmp = (img_as_np[c] - mean_np[c])**2
            std_np[c] += float(np.sum(tmp))
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)