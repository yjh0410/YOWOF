from .backbone_2d.resnet import build_resnet_2d
from .backbone_3d.resnet import build_resnet_3d


def build_backbone_2d(model_name='resnet18', pretrained=False):
    """ Build ResNet"""
    print('==============================')
    print('2D Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet_2d(
            model_name=model_name,
            pretrained=pretrained,
            )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim


def build_backbone_3d(model_name='resnet18', pretrained=False, part=False):
    """ Build ResNet"""
    print('==============================')
    print('3D Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet_3d(
            model_name=model_name,
            pretrained=pretrained,
            part=part
            )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
