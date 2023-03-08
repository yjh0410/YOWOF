from .resnet import build_resnet
from .dla import build_dla
from .vgg import build_vgg


def build_backbone(cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(cfg['backbone'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone'] in ['resnet18', 'resnet50', 'resnet101', 'resnext101']:
        model, feat_dim = build_resnet(
            model_name=cfg['backbone'],
            pretrained=pretrained,
            norm_layer=cfg['norm_layer'],
            res5_dilation=cfg['res5_dilation'])

    elif cfg['backbone'] in ['dla34']:
        model, feat_dim = build_dla(model_name=cfg['backbone'], pretrained=cfg['pretrained'])

    elif cfg['backbone'] in ['vgg16']:
        model, feat_dim = build_vgg(model_name=cfg['backbone'], pretrained=cfg['pretrained'])

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
