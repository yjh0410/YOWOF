from .resnet import build_resnet_2d


def build_backbone(model_name='resnet18', pretrained=False, res5_dilation=False):
    print('==============================')
    print('2D Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet_2d(model_name, pretrained, res5_dilation)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
