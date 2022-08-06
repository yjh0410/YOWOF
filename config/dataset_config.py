# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        # 'data_root': 'E:/python_work/spatial-temporal_action_detection/dataset/UCF24',
        'anno_file': 'UCF101v2-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # freeze
        'freeze_backbone': False,
        # train config
        'batch_size': 16,
        'accumulate': 1,
        'len_clip': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'max_epoch': 10,
        'lr_epoch': [2, 3, 4, 5],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # anchor size
        'anchor_size': [[32., 54.],
                        [57., 116.], 
                        [72., 186.],
                        [105., 225.],
                        [161., 270.]] # 320
    },
    
    'jhmdb': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/JHMDB',
        # 'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/JHMDB',
        'anno_file': 'JHMDB-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # freeze
        'freeze_backbone': False,
        # train config
        'batch_size': 16,
        'accumulate': 8,
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
        'anchor_size': [[43., 141.],
                        [76., 183.],
                        [80., 258.],
                        [141., 265.],
                        [224., 285.]]

    },
    
    'ava':{
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'JHMDB-GT.pkl',
        # loss
    }
}