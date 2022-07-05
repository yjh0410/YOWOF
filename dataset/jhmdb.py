import numpy as np
import os
import cv2
import pickle

import torch
import torch.utils.data as data

torch.multiprocessing.set_sharing_strategy('file_system')

try:
    from utils import tubelet_in_out_tubes, tubelet_has_gt
except:
    from .utils import tubelet_in_out_tubes, tubelet_has_gt


JHMDB_CLASSES = (
    'brush_hair',   'catch',          'clap',        'climb_stairs',
    'golf',         'jump',           'kick_ball',   'pick', 
    'pour',         'pullup',         'push',        'run',
    'shoot_ball',   'shoot_bow',      'shoot_gun',   'sit',
    'stand',        'swing_baseball', 'throw',       'walk',
    'wave'
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class JHMDB(data.Dataset):
    def __init__(self, 
                 cfg,
                 img_size=224, 
                 len_clip=1,
                 is_train=False,
                 transform=None,
                 debug=False
                 ):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.len_clip = len_clip
        self.root = cfg['data_root']
        self.anno_file = cfg['anno_file']
        self.image_path = os.path.join(self.root, 'Frames')
        self.flow_path = os.path.join(self.root, 'FlowBrox04')
        self.transform = transform
        self.is_train = is_train
        self.debug = debug
        if is_train:
            self.split = cfg['train_split']
        else:
            self.split = cfg['test_split']

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

        # basic attributes
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
        self.video_list = video_list
        self.num_videos = len(video_list)
        
        if self.debug:
            video_list = video_list[:100]

        for v in video_list:
            vtubes = sum(self.gttubes[v].values(), [])
            for i in range(1, self.nframes[v] + 2 - self.len_clip):
                flag_1 = tubelet_in_out_tubes(vtubes, i, self.len_clip)
                flag_2 = tubelet_has_gt(vtubes, i, self.len_clip)
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
        for i in range(self.len_clip):
            cur_fid = frame + i
            # load an image
            image_list[i] = self.load_single_image(video_name, cur_fid)

        return image_list


    def load_video(self, index):
        if self.is_train:
            video_list = self.train_videos[self.split - 1]
        else:
            # get test video list
            video_list = self.test_videos[self.split - 1]
        
        return video_list[index] # video name


    def pull_item(self, index):
        video_name, frame = self.indices[index]
        image_list = []
        target_list = []
        for i in range(self.len_clip):
            cur_fid = frame + i
            # load an image
            image_file = os.path.join(self.image_path, video_name, 
                                        '{:0>5}.png'.format(cur_fid))
            image_list.append(cv2.imread(image_file))

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
            target_list.append(cur_target_list)

        # augment
        if self.transform:
            image_list, target_list = self.transform(image_list, target_list)
        # image_list = [image_1, 
        #               image_2, 
        #               ..., 
        #               image_K]
        # target_list = [ndarray([[x1, y1, x2, y2, cls, fid], 
        #                             ...]),
        #                ndarray([[x1, y1, x2, y2, cls, fid], 
        #                             ...]]
        return image_list, target_list


if __name__ == '__main__':
    from transforms import TrainTransforms, ValTransforms
    dataset_config={
        # dataset
        'data_root': 'E:/python_work/spatial-temporal_action_detection/dataset/JHMDB',
        'anno_file': 'JHMDB-GT.pkl',
        'train_split': 1,
        'test_split': 1,
    }
    len_clip = 7
    is_train=True,

    img_size=224
    format = 'RGB'
    pixel_mean=(123.675, 116.28, 103.53), 
    pixel_std=(58.395, 57.12, 57.375),
    trans_config = [{'name': 'DistortTransform',
                     'hue': 0.1,
                     'saturation': 1.5,
                     'exposure': 1.5},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 16},
                    {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                    {'name': 'ToTensor'},
                    {'name': 'Resize'},
                    {'name': 'Normalize'}]

    transform=TrainTransforms(trans_config=trans_config,
                              img_size=img_size,
                              format='RGB')


    dataset = JHMDB(cfg=dataset_config, 
                    img_size = img_size,
                    len_clip = len_clip,
                    is_train = is_train,
                    transform = transform,
                    debug=True)

    for i in range(100):
        image_list, target_list = dataset[i]

        # vis images
        for idx in range(len_clip):
            image = image_list[idx]
            # to numpy
            image = image.permute(1, 2, 0).numpy()
            # to BGR format
            if format == 'RGB':
                # denormalize
                image = image * pixel_std + pixel_mean
                image = image[:, :, (2, 1, 0)].astype(np.uint8)
            image = image.copy()
            target = target_list[idx]
            for tgt in target:
                x1, y1, x2, y2, cls_id, fid = tgt
                label = JHMDB_CLASSES[int(cls_id)]
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

