# Model Configuration


yowof_config = {
    'yowof-r18': {
        # backbone
        'backbone': 'resnet18',
        'pretrained': True,
        'res5_dilation': False,
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_nl': 2,
        'conv_lstm_di': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.3,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # batch size
        'batch_size': 8,
        'accumulate': 16,
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
        # anchor size
        'anchor_size': [[16, 16],
                        [32, 32],
                        [64, 64],
                        [128, 128],
                        [256, 256]], # 320
    },

    'yowof-r50': {
        # backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'res5_dilation': False,
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_nl': 2,
        'conv_lstm_di': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.3,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # batch size
        'batch_size': 8,
        'accumulate': 16,
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
        # anchor size
        'anchor_size': [[16, 16],
                        [32, 32],
                        [64, 64],
                        [128, 128],
                        [256, 256]], # 320
    },

    'yowof-r50-DC5': {
        # backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'res5_dilation': True,
        'stride': 16,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_nl': 2,
        'conv_lstm_di': 2,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.3,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # batch size
        'batch_size': 8,
        'accumulate': 16,
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
        # anchor size
        'anchor_size': [[16, 16],
                        [32, 32],
                        [64, 64],
                        [128, 128],
                        [256, 256]], # 320
    },

    'yowof-r101': {
        # backbone
        'backbone': 'resnet101',
        'pretrained': True,
        'res5_dilation': False,
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_nl': 2,
        'conv_lstm_di': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.3,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # batch size
        'batch_size': 8,
        'accumulate': 16,
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
        # anchor size
        'anchor_size': [[16, 16],
                        [32, 32],
                        [64, 64],
                        [128, 128],
                        [256, 256]], # 320
    },

}