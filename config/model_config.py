# Model configuration


model_config = {
    'yowof-d19': {
        # input
        'train_size': 320,
        'test_size': 320,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],  # imagenet pixel mean
        'pixel_std': [58.395, 57.12, 57.375],     # imagenet pixel std
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'RandomShift', 'max_shift': 32},
                         {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # backbone
        ## 2D
        'backbone_2d': 'yolov2',
        'pretrained_2d': True,
        'stride': 32,
        # conv lstm
        'conv_lstm_hdm': 512,
        'conv_lstm_ks': 3,
        'conv_lstm_nl': 2,
        # head
        'head_dim': 1024,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,

    },

}