# predicted as a batch
import os
import glob
import numpy as np
from PIL import Image
import scipy.io as sio
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from params import par
from model import DeepVIO
from data_helper import get_data_info_test
from helper import eulerAnglesToRotationMatrix, angle_to_R, normalize_angle_delta
from evaluation import kittiOdomEval


torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0')

def test(path_list):
    # Path
    save_dir = f'result/{par.experiment_name}/'  # directory to save prediction answer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_len = 41

    # Load model
    VIONet = DeepVIO(par)

    VIONet.load_state_dict(torch.load(par.load_ckpt, map_location='cpu'))
    VIONet = VIONet.cuda(device)

    print('Load model from: ', par.load_ckpt)

    x_scaling, y_scaling = 1.1, 1.1
    scaled_h, scaled_w = int(par.img_h * y_scaling), int(par.img_w * x_scaling)

    VIONet.eval()
    for test_video in path_list:
        df = get_data_info_test(folder=test_video, seq_len=test_len)
        image_arr = np.asarray(df.image_path)  # image paths
        groundtruth_arr = np.asarray(df.pose)
        imu_arr = np.asarray(df.imu)

        prev = None
        answer = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], ]
        pose_est_list = []
        ang_err_list = []
        trans_err_list = []
        pose_gt_list = []
        yaw_list = []

        for i in range(len(df)):

            print('{} / {}'.format(i, len(df)), end='\r', flush=True)
            # Load the test images
            image_path_sequence = image_arr[i]
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

            image_sequence = torch.cat(image_sequence, 0).to(device)
            image_sequence_LR = torch.cat(image_sequence_LR, 0).to(device)

            imu_sequence = torch.FloatTensor(imu_arr[i])

            gt_sequence = groundtruth_arr[i][:, :6]
            
            with torch.no_grad():
                x_in1 = image_sequence.unsqueeze(0)
                i_in = imu_sequence.to(device).unsqueeze(0)
                angle, trans, hc = VIONet(x_in1, i_in, prev=prev)
                prev = hc         

            angle = angle.squeeze().detach().cpu().numpy()
            trans = trans.squeeze().detach().cpu().numpy()

            pose_pred = np.hstack((angle, trans))

            # Record the estimation error
            ang_err_list.append(np.mean((gt_sequence[:, :3] - angle)**2))
            trans_err_list.append(np.mean((gt_sequence[:, 3:] - trans)**2))
            yaw_list.append(np.mean((gt_sequence[:, 1] - angle[:, 1])**2))

            pose_est_list.append(pose_pred)
            pose_gt_list.append(gt_sequence)

            # Accumulate the relative poses
            for index in range(angle.shape[0]):
                poses_pre = answer[-1]
                poses_pre = np.array(poses_pre).reshape(3, 4)
                R_pre = poses_pre[:, :3]
                t_pre = poses_pre[:, 3]

                pose_rel = pose_pred[index, :]
                Rt_rel = angle_to_R(pose_rel)
                R_rel = Rt_rel[:, :3]
                t_rel = Rt_rel[:, 3]

                R = R_pre @ R_rel
                t = R_pre.dot(t_rel) + t_pre

                pose = np.concatenate((R, t.reshape(3, 1)), 1).flatten().tolist()
                answer.append(pose)
        
        ang_err_m = np.mean(ang_err_list)
        trans_err_m = np.mean(trans_err_list)
        yaw_err_m = np.mean(yaw_list)

        poses_rel_est = np.vstack(pose_est_list)
        poses_rel_gt = np.vstack(pose_gt_list)
        
        sio.savemat(test_video + '.mat', {'poses_est': poses_rel_est, 'poses_gt': poses_rel_gt})

        # Save answer
        with open('{}/{}_pred.txt'.format(save_dir, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(' '.join([str(r) for r in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')


if __name__ == '__main__':

    test(['05', '07', '10'])

    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--gt_dir', type=str, default=par.pose_dir, help='Directory path of the ground truth odometry')
    parser.add_argument('--result_dir', type=str, default=f'./result/{par.experiment_name}/', help='Directory path of storing the odometry results')
    parser.add_argument('--eva_seqs', type=str, default='05_pred,07_pred,10_pred', help='The sequences to be evaluated')
    parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to convert the pose to camera coordinate')

    args = parser.parse_args()
    pose_eval = kittiOdomEval(args)
    pose_eval.eval(toCameraCoord=args.toCameraCoord)   # set the value according to the predicted results