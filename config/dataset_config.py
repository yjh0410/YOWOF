# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/ucf24',
        # 'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/ucf24',
        # input size
        'train_size': 320,
        'test_size': 320,
        # transform
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # train config
        'batch_size': 16,
        'accumulate': 8,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                )
    },
    
    'jhmdb21': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/jhmdb21',
        'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/jhmdb21',
        # input size
        'train_size': 320,
        'test_size': 320,
        # transform
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # train config
        'batch_size': 16,
        'accumulate': 1,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'max_epoch': 12,
        'lr_epoch': [8, 10],
        'base_lr': 2e-5,
        'lr_decay_ratio': 0.1,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'label_map': (
                    'brush_hair',   'catch',          'clap',        'climb_stairs',
                    'golf',         'jump',           'kick_ball',   'pick', 
                    'pour',         'pullup',         'push',        'run',
                    'shoot_ball',   'shoot_bow',      'shoot_gun',   'sit',
                    'stand',        'swing_baseball', 'throw',       'walk',
                    'wave'
                )
    },
    
    'ava':{
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'JHMDB-GT.pkl',
        # loss
    }
}