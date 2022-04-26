import numpy as np
import os
import random
import cv2
import pickle

import torch.utils.data as data

try:
    from ACT_utils import tubelet_in_out_tubes, tubelet_has_gt
except:
    from .ACT_utils import tubelet_in_out_tubes, tubelet_has_gt


UCF24_CLASSES = (
    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
)


class UCF24(data.Dataset):
    def __init__(self, 
                 cfg,
                 img_size=224, 
                 data_dir=None, 
                 anno_file=None,
                 split=None,
                 is_train=False,
                 transform=None,
                 debug=False
                 ):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.root = data_dir
        self.image_path = os.path.join(self.root, 'rgb-images')
        self.flow_path = os.path.join(self.root, 'brox-images')
        self.anno_file = anno_file
        self.transform = transform
        self.split = split
        self.is_train = is_train
        self.debug = debug

        self.load_datas()


    def load_datas(self):
        print('loading data ...')
        # laod anno pkl file
        # pkl is dict value:
        # 'labels': [label, ...]
        # 'gttubes': {'video_name_1': {'action_label': [[frame_id, x1, y1, x2, y2], ...],
        #                              ...},
        #              ...}
        # 'nframes': {'video_name_1': nframes,
        #             ...}
        # 'train_videos': ['train_video_name_1', 
        #                  ...]
        # 'test_videos': ['test_video_name_1', 
        #                 ...]
        # 'resolution': {'video_name_1': resolution,
        #                ...}
        pkl_file = os.path.join(self.root, self.anno_file)
        with open(pkl_file, 'rb') as fid:
            pkl = pickle.load(fid, encoding='iso-8859-1')

        # basic attributes of UCF24 dataset
        self.nframes = pkl['nframes']
        self.gttubes = pkl['gttubes']
        self.labels = pkl['labels']
        self.train_videos = pkl['train_videos']
        self.test_videos = pkl['test_videos']
        self.resolution = pkl['resolution']

        indices = []
        if self.is_train:
            # get train video list
            video_list = self.train_videos[self.split - 1]
        else:
            # get test video list
            video_list = self.test_videos[self.split - 1]

        if self.debug:
            video_list = video_list[:100]

        for v in video_list:
            vtubes = sum(self.gttubes[v].values(), [])
            for i in range(1, self.nframes[v] + 2 - self.cfg['K']):
                flag_1 = tubelet_in_out_tubes(vtubes, i, self.cfg['K'])
                flag_2 = tubelet_has_gt(vtubes, i, self.cfg['K'])
                if flag_1 and flag_2:
                    indices += [(v, i)]
        print('loading done !')

        self.indices = indices


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        return self.pull_item(index)


    def load_single_image(self, video_name, frame_id):
        # load an image
        image_file = os.path.join(self.image_path, video_name, 
                                    '{:0>5}.jpg'.format(frame_id))
        image = cv2.imread(image_file)

        return image


    def load_video_clip(self, index):
        video_name, frame = self.indices[index]
        image_list = {}
        for i in range(self.cfg['K']):
            cur_fid = frame + i
            # load an image
            image_list[i] = self.load_single_image(video_name, cur_fid)

        return image_list


    def pull_item(self, index):
        video_name, frame = self.indices[index]
        image_list = {}
        target_list = {}
        for i in range(self.cfg['K']):
            cur_fid = frame + i
            # load an image
            image_file = os.path.join(self.image_path, video_name, 
                                        '{:0>5}.jpg'.format(cur_fid))
            image_list[i] = cv2.imread(image_file)

            # load a target
            cur_vid_gttube = self.gttubes[video_name]
            cur_target_list = []
            for label, tubes in cur_vid_gttube.items():
                for tube in tubes:
                    if cur_fid not in tube[:, 0]:
                        continue
                    else:
                        tube = tube.copy()
                        idx = (cur_fid == tube[:, 0])
                        gt_boxes = tube[idx, 1:5][0]
                        gt_label = label
                        cur_target_list.append([*gt_boxes, gt_label, cur_fid])

            # check target
            if len(cur_target_list) == 0:
                cur_target_list.append([])
            # to ndarray
            cur_target_list = np.array(cur_target_list).reshape(-1, 6)
            target_list[i] = cur_target_list

        # image_list = {0: image_1, 
        #               1: image_2, 
        #               ..., 
        #               K: image_K]
        # target_list = {0:[[x1, y1, x2, y2, cls, fid], 
        #                     ...],
        #                K:[[x1, y1, x2, y2, cls, fid], 
        #                     ...]}
        return image_list, target_list


if __name__ == '__main__':
    dataset_config={'K': 3}
    img_size=224,
    data_dir='E:/python_work/spatial-temporal_action_detection/dataset/UCF24'
    anno_file='UCF101v2-GT.pkl'
    split=1
    is_train=True,
    transform=None

    dataset = UCF24(cfg=dataset_config, 
                    img_size = img_size,
                    data_dir = data_dir,
                    anno_file = anno_file,
                    split = split,
                    is_train = is_train,
                    transform = transform,
                    debug=True)

    for i in range(100):
        image_list, target_list = dataset[i]

        # vis images
        for idx in range(dataset_config['K']):
            image = image_list[idx].copy()
            target = target_list[idx]
            for tgt in target:
                x1, y1, x2, y2, cls_id, fid = tgt
                label = UCF24_CLASSES[int(cls_id)]
                # put the test on the bbox
                cv2.putText(image, 
                            label, 
                            (int(x1), int(y1 - 5)), 
                            0, 0.5, (255, 0, 0), 1, 
                            lineType=cv2.LINE_AA)

                cv2.rectangle(image,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (255, 0, 0), 2)
            cv2.imshow('image', image)
            cv2.waitKey(0)

