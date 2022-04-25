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
        # 'labels': 
        # 'gttubes': {'video_name_1': 
        #                   {'action_label': [[frame_id, x1, y1, x2, y2], ...],
        #                    'action_label': [[frame_id, x1, y1, x2, y2], ...],
        #                    ...},
        #             'video_name_2': 
        #                   {'action_label': [[frame_id, x1, y1, x2, y2], ...],
        #                    'action_label': [[frame_id, x1, y1, x2, y2], ...],
        #                    ...},
        #               ...
        #               }
        # 'nframes': {'video_name_1': nframes,
        #             'video_name_2': nframes,
        #             ...}
        # 'train_videos': ['train_video_name_1', 
        #                  'train_video_name_2',
        #                  ...]
        # 'test_videos': ['test_video_name_1', 
        #                 'test_video_name_2',
        #                 ...]
        # 'resolution': {'video_name_1': resolution,
        #                'video_name_2': resolution,
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


    def load_image(self, img_id):
        return


    def pull_item(self, index):
        video_name, frame = self.indices[index]
        image_list = []
        for i in range(self.cfg['K']):
            image_file = os.path.join(self.image_path, video_name, '{:0>5}.jpg'.format(frame + i))
            image_list.append(cv2.imread(image_file))
        orig_h, orig_w = self.resolution[video_name]
        target_list = {}
        for label, tubes in self.gttubes[video_name].items():
            for t in tubes:
                if frame not in t[:, 0]:
                    continue
                assert frame + self.cfg['K'] - 1 in t[:, 0]
                t = t.copy()
                mask = (t[:, 0] >= frame) * (t[:, 0] < frame + self.cfg['K'])
                gt_boxes = t[mask, 1:5]

                assert gt_boxes.shape[0] == self.cfg['K']

                if label not in target_list:
                    target_list[label] = []
                # gt_bbox[ilabel] ---> a list of numpy array, each one is K, x1, x2, y1, y2
                target_list[label].append(gt_boxes)

        return image_list, target_list


if __name__ == '__main__':
    dataset_config={'K': 7
    }
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
        for fi, img in enumerate(image_list):
            img = img.copy()
            for cls_id in target_list.keys():
                tubes = target_list[cls_id]
                for tube in tubes:
                    gt_box = tube[fi]
                    x1, y1, x2, y2 = gt_box
                    cv2.rectangle(img,
                                 (int(x1), int(y1)),
                                 (int(x2), int(y2)),
                                 (255, 0, 0))
            cv2.imshow('image', img)
            cv2.waitKey(0)

