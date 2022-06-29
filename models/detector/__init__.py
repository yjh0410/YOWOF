import torch
from .yowof.yowof import YOWOF


# build YOWOF detector
def build_model(args, 
                cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                inference='clip',
                coco_pretrained=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    if 'yowof' in args.version:
        model = YOWOF(cfg=cfg,
                      device=device,
                      img_size=cfg['train_size'] if trainable else cfg['test_size'],
                      num_classes=num_classes, 
                      trainable=trainable,
                      conf_thresh=cfg['conf_thresh'],
                      nms_thresh=cfg['nms_thresh'],
                      topk=args.topk)

    # set inference mode
    if not trainable:
        if inference == 'clip':
            model.stream_infernce = False
        elif inference == 'stream':
            model.stream_infernce = True

    # Load COCO pretrained weight
    if coco_pretrained is not None:
        print('Loading COCO pretrained weight ...')
        checkpoint = torch.load(coco_pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    print(k)
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)
                        
    return model
