# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'UCF101v2-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # train config
        'max_epoch': 12,
        'batch_size': 32,
        'base_lr': 0.001,
        'min_lr_ratio': 0.05,
        'no_cos_decay': 5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,

    },
    'jhmdb': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/JHMDB',
        'anno_file': 'JHMDB-GT.pkl',
        'train_split': 1,
        'test_split': 1,
        # train config
        'max_epoch': 12,
        'batch_size': 32,
        'base_lr': 0.001,
        'min_lr_ratio': 0.05,
        'no_cos_decay': 5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,

    },
    'ava':{
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'JHMDB-GT.pkl',
        # loss
    }
}