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
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'max_epoch': 12,
        'lr_epoch': [6, 10],
        'batch_size': 16,
        'accumulate': 1,
        'base_lr': 2.5e-5,
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
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'max_epoch': 12,
        'lr_epoch': [6, 10],
        'batch_size': 16,
        'accumulate': 1,
        'base_lr': 2.5e-5,
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