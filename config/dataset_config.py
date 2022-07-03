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
        'optimizer': 'sgd',
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'max_epoch': 10,
        'lr_epoch': [3, 5, 7],
        'batch_size': 128,
        'accumulate': 1,
        'base_lr': 0.01 / 32.,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
    },
    
    'jhmdb': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/JHMDB',
        # 'data_root': 'E:/python_work/spatial-temporal_action_detection/dataset/JHMDB',
        'anno_file': 'JHMDB-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # freeze
        'freeze_backbone': False,
        # train config
        'optimizer': 'sgd',
        'weight_decay': 0,
        'momentum': 0.9,
        'max_epoch': 10,
        'lr_epoch': [3, 5, 7],
        'batch_size': 16,
        'accumulate': 2,
        'base_lr': 0.01 / 32.,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,

    },
    
    'ava':{
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'JHMDB-GT.pkl',
        # loss
    }
}