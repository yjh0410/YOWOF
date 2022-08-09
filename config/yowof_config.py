# Model Configuration


yowof_config = {
    'yowof-r18': {
        # backbone
        'backbone_2d': 'resnet18',
        'pretrained_2d': True,
        'res5_dilation': False,
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_nl': 2,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.6,
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
        'loss_reg_weight': 3.0,

    },
}