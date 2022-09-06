from .resnet import build_resnet_2d


def build_backbone(model_name='resnet18', pretrained=False, res5_dilation=False):
    print('==============================')
    print('2D Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if model_name in ['resnet18', 'resnet50', 'resnet101', 'resnext101']:
        model, feat_dim = build_resnet_2d(model_name, pretrained, res5_dilation)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
