import torch
from thop import profile


def FLOPs_and_Params(model, img_size, len_clip, device):
    # set eval mode
    model.trainable = False
    model.eval()

    # initalize model
    model.initialization = True
    model.set_inference_mode(mode='stream')

    # generate init video clip
    video_clip = torch.randn(1, len_clip, 3, img_size, img_size).to(device)
    outputs = model(video_clip)

    # generate a new frame
    video_clip = torch.randn(1, len_clip, 3, img_size, img_size).to(device)

    print('==============================')
    flops, params = profile(model, inputs=(video_clip, ))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    
    # set train mode.
    model.trainable = True
    model.train()


if __name__ == "__main__":
    pass
