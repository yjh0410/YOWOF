# Model Configuration


yowof_config = {
    'yowof-r18': {
        # backbone
        'backbone': 'resnet18',
        'pretrained': True,  # pretrained on the kinetic-400
        'res5_dilation': False,
        'norm_layer': 'BN',
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_pd': 1,
        'conv_lstm_di': 1,
        'conv_lstm_nl': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
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
        'pretrained': True,  # pretrained on the kinetic-400
        'res5_dilation': False,
        'norm_layer': 'BN',
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_pd': 1,
        'conv_lstm_di': 1,
        'conv_lstm_nl': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
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

    'yowof-vgg16': {
        # backbone
        'backbone': 'vgg16',
        'pretrained': True,  # pretrained on the imagenet
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_pd': 1,
        'conv_lstm_di': 1,
        'conv_lstm_nl': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
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

    'yowof-dla34': {
        # backbone
        'backbone': 'dla34',
        'pretrained': True,  # pretrained on the imagenet
        'stride': 32,
        # temporal encoder
        'conv_lstm_ks': 3,
        'conv_lstm_pd': 1,
        'conv_lstm_di': 1,
        'conv_lstm_nl': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'relu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
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