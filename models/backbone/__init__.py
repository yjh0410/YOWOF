from .resnet import build_resnet
from .shufflenetv2 import build_sfnetv2


def build_backbone(model_name='resnet50-d', 
                   pretrained=False, 
                   norm_type='BN',
                   freeze=False):
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet(model_name=model_name, 
                                       pretrained=pretrained,
                                       norm_type=norm_type)

    elif 'shufflenet' in model_name:
        model, feat_dim = build_sfnetv2(pretrained=pretrained)
        
    else:
        print('Unknown Backbone ...')
        exit()

    if freeze:
        print('freeze parameters backone ...')
        for m in model.parameters():
            m.requires_grad = False

    return model, feat_dim
