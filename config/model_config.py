# Model configuration


model_config = {
    'baseline': {
        # input
        'train_size': 320,
        'test_size': 320,
        'len_clip': 1,
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
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'spp_block',
        'pooling_size': [5, 9, 13],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[8, 8], 
                        [16, 16],
                        [32, 32], 
                        [64, 64], 
                        [128, 128]],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'optimizer': 'sgd',
        'weight_decay': 1e-4,
        'momentum': 0.9,
    },

    'yowof-r18': {
        # input
        'train_size': 320,
        'test_size': 320,
        'len_clip': 7,
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
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'spp_block',
        'pooling_size': [5, 9, 13],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[8, 8], 
                        [16, 16],
                        [32, 32], 
                        [64, 64], 
                        [128, 128]],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'optimizer': 'sgd',
        'weight_decay': 1e-4,
        'momentum': 0.9,

    },

    'yowof-r50': {
        # input
        'img_size': 320,
        'len_clip': 7,
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
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'spp_block',
        'pooling_size': [5, 9, 13],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[8, 8], 
                        [16, 16],
                        [32, 32], 
                        [64, 64], 
                        [128, 128]],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },

    'yowof-r50-D': {
        # input
        'img_size': 320,
        'len_clip': 7,
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
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'pretrained': None,
        # neck
        'neck': 'spp_block',
        'pooling_size': [5, 9, 13],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [8, 16, 32, 64, 128, 256],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },

    'yowof-sfv2': {
        # input
        'img_size': 320,
        'len_clip': 7,
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
        # model
        'backbone': 'shufflenetv2',
        'norm_type': '',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'spp_block',
        'pooling_size': [5, 9, 13],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # head
        'head_dim': 128,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': True,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[8, 8], 
                        [16, 16],
                        [32, 32], 
                        [64, 64], 
                        [128, 128]],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },

}