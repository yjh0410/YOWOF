from .resnet import build_resnet
from .shufflenetv2 import build_sfnetv2


def build_backbone(model_name='resnet18', pretrained=False, res5_dilation=False):
    """ Build ResNet"""
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))
    print('--res5_dilation: {}'.format(res5_dilation))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet(
            model_name=model_name,
            pretrained=pretrained,
            res5_dilation=res5_dilation
            )

    elif 'shufflenet' in model_name:
        model, feat_dim = build_sfnetv2(pretrained=pretrained)
        
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
