import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = ['vgg16']


model_urls = {
    'vgg16': 'https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/vgg16_coco.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

        # freeze
        self._freeze()


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # freeze
        self._freeze()


    def _freeze(self):
        for name, parameter in self.named_parameters():
            if name in ["features.0.", "features.2.", "features.5.", "features.7."]:
                parameter.requires_grad_(False)


    def forward(self, x):
        for m in self.features:
            x = m(x)

        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        print('Load pretrained weight for backbone {}'.format(model_urls[arch]))
        # checkpoint state dict
        checkpoint = load_state_dict_from_url(
            model_urls[arch],
            progress=progress
            )
        checkpoint_state_dict = checkpoint.pop('model')

        # model state dict
        model_state_dict = model.state_dict()

        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


# build dla
def build_vgg(model_name='vgg16', pretrained=False):
    if model_name == 'vgg16':
        model = vgg16(pretrained)
        feat = 512

    return model, feat


if __name__ == '__main__':
    import time

    # cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # build resnet
    model, feat_dim = build_vgg('vgg16', pretrained=True)
    model = model.to(device)

    # test
    x = torch.ones(1, 3, 64, 64).to(device)
    for i in range(1):
        # star time
        t0 = time.time()
        y = model(x)
        print(y.size())
        print('time', time.time() - t0)
