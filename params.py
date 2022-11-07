import os


class Parameters():
    def __init__(self):
        self.devices = [0]
        self.n_processors = 4
        # Path
        self.data_dir = './data'
        self.image_dir = self.data_dir + '/sequences/'
        self.pose_dir = self.data_dir + '/poses/'
        self.imu_dir = self.data_dir + '/imus/'

        self.train_video = ['00', '01', '02', '04', '06', '08', '09']
        self.valid_video = ['05', '07', '10']

        self.imu_per_image = 10
        self.imu_int_prev = 0

        self.experiment_name = 'low_latency'

        # Data Preprocessing
        self.img_w = 512   # original size is about 1226
        self.img_h = 256   # original size is about 370

        # Data Augmentation
        self.is_hflip = True
        self.is_color = False
        self.flag_imu_aug = False
        self.is_crop = False

        self.rnn_hidden_size = 1024
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.2
        self.rnn_dropout_between = 0.2   # 0: no dropout
        self.clip = None
        self.batch_norm = True

        self.imu_method = 'conv'  # ['bi-LSTM', 'conv']
        self.imu_hidden_size = 128
        self.fuse_method = 'cat'  # ['cat', 'soft']
        self.visual_f_len = 512
        self.imu_f_len = 256

        self.dropout = 0
        self.imu_prev = 0

        # Training
        self.decay = 5e-6
        self.batch_size = 16
        self.pin_mem = True

        # Select searched model
        self.target = 'flops'
        if self.target == 'flops':
            self.load_ckpt = './flops_target.ckpt'
        else:
            self.load_ckpt = './latency_target.ckpt'


par = Parameters()
