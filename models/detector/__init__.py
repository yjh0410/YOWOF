import torch
from .yowof.yowof import YOWOF


# build YOWOF detector
def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                inference='clip',
                coco_pretrained=None,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # Basic config
    if trainable:
        img_size = m_cfg['train_size']
    else:
        img_size = m_cfg['test_size']

    # build YOWOF
    if 'yowof' in args.version:
        model = YOWOF(
            cfg=m_cfg,
            device=device,
            anchor_size=d_cfg['anchor_size'],
            img_size=img_size,
            len_clip=d_cfg['len_clip'],
            num_classes=num_classes,
            conf_thresh=m_cfg['conf_thresh'],
            nms_thresh=m_cfg['nms_thresh'],
            topk=args.topk,
            trainable=trainable
            )

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

    # Freeze backbone
    if d_cfg['freeze_backbone']:
        print('Freeze Backbone ...')
        for m in model.backbone.parameters():
            m.requires_grad = False

    # keep training       
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
