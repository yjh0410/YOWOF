# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        # 'data_root': 'E:/python_work/spatial-temporal_action_detection/dataset/UCF24',
        'anno_file': 'UCF101v2-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # input
        'train_size': 224,
        'test_size': 224,
        'format': 'RGB',
        'pixel_mean': [0., 0., 0.],          # imagenet pixel mean
        'pixel_std': [255., 255., 255.],     # imagenet pixel std
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'JitterCrop', 'jitter_ratio': 0.2},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # train config
        'batch_size': 8,
        'accumulate': 16,
        'len_clip': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 10,
        'lr_epoch': [2, 3, 4, 5],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # anchor size
        'anchor_size': [[22, 38],
                        [40, 81],
                        [51, 130],
                        [73, 158],
                        [112, 189]] # 224
    },
    
    'jhmdb': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/JHMDB',
        # 'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/JHMDB',
        'anno_file': 'JHMDB-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # input
        'train_size': 224,
        'test_size': 224,
        'format': 'RGB',
        'pixel_mean': [0., 0., 0.],          # imagenet pixel mean
        'pixel_std': [255., 255., 255.],     # imagenet pixel std
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # freeze backbone
        'freeze_backbone_2d': True,
        'freeze_backbone_3d': True,
        # train config
        'batch_size': 8,
        'accumulate': 16,
        'len_clip': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 10,
        'lr_epoch': [3, 4, 5, 6],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # anchor size
        'anchor_size': [[30,  99],
                        [53, 128],
                        [56, 180],
                        [98, 185],
                        [157, 200]] # 224

    },
    
    'ava':{
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'JHMDB-GT.pkl',
        # loss
    }
}