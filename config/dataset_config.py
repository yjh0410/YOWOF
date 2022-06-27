# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        # 'data_root': 'E:/python_work/spatial-temporal_action_detection/dataset/UCF101_24',
        'anno_file': 'UCF101v2-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # train config
        'max_epoch': 12,
        'lr_epoch': [6, 10],
        'batch_size': 32,
        'base_lr': 0.012/64.,
        'bk_lr_ratio': 1.0/3.0,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,

    },
    
    'jhmdb': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/JHMDB',
        'data_root': 'E:/python_work/spatial-temporal_action_detection/dataset/JHMDB',
        'anno_file': 'JHMDB-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # train config
        'max_epoch': 8,
        'lr_epoch': [4, 8],
        'batch_size': 32,
        'base_lr': 0.12/64.,
        'bk_lr_ratio': 1.0/3.0,
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