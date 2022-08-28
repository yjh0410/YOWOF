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
        'head_act': 'lrelu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # anchor size
        'anchor_size': {
            'ucf24': [[32, 54],
                      [57, 116],
                      [72, 186],
                      [105, 225],
                      [161, 270]], # 224
            'jhmdb21': [[43,  141],
                        [76, 183],
                        [80, 258],
                        [141, 265],
                        [224, 285]], # 224
            'ava_v2.2': [[33, 97],
                         [58, 189],
                         [97, 233],
                         [149, 268],
                         [236, 290]] # 224
                         }},

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
        'head_act': 'lrelu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
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
        'conv_lstm_di': 1,
        # head
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_heads': 4,
        'num_reg_heads': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'conf_thresh_valid': 0.005,
        'nms_thresh': 0.5,
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
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