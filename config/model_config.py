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
                         {'name': 'PadImage'}],
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [1, 2, 3, 4],
        'expand_ratio': 0.25,
        'act_type': 'relu',
        'neck_norm': 'BN',
        # head
        'head_dim': 512,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_head': 2,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_nms_thresh': 0.45,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_size': [8, 16, 32, 64, 128],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },
    
    'yowof18': {
        # input
        'img_size': 320,
        'len_clip': 7,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'RandomShift', 'max_shift': 32},
                         {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [1, 2, 3, 4],
        'expand_ratio': 0.25,
        'act_type': 'relu',
        'neck_norm': 'BN',
        # head
        'head_dim': 512,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_head': 2,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_nms_thresh': 0.45,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_size': [8, 16, 32, 64, 128],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },

    'yowof50': {
        # input
        'img_size': 320,
        'len_clip': 7,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'RandomShift', 'max_shift': 32},
                         {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'pretrained': None,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [1, 2, 3, 4],
        'expand_ratio': 0.25,
        'act_type': 'relu',
        'neck_norm': 'BN',
        # head
        'head_dim': 512,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_head': 2,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_nms_thresh': 0.45,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_size': [8, 16, 32, 64, 128],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },

    'yowof50-D': {
        # input
        'img_size': 320,
        'len_clip': 7,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'RandomShift', 'max_shift': 32},
                         {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'pretrained': None,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        'act_type': 'relu',
        'neck_norm': 'BN',
        # head
        'head_dim': 512,
        'head_norm': 'BN',
        'act_type': 'relu',
        'num_cls_head': 2,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_nms_thresh': 0.45,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_size': [8, 16, 32, 64, 128, 256],
        # matcher
        'matcher': 'uniform_matcher',
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,

    },

}