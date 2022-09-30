from .resnet import build_resnet_2d


def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(cfg['backbone'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone'] in ['resnet18', 'resnet50', 'resnet101', 'resnext101']:
        model, feat_dim = build_resnet_2d(
            model_name=cfg['backbone'],
            pretrained=pretrained,
            norm_layer=cfg['norm_layer'],
            res5_dilation=cfg['res5_dilation'])

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
