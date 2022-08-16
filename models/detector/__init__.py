import torch
from .yowof.yowof import YOWOF


# build YOWO detector
def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # Basic config
    if trainable:
        img_size = d_cfg['train_size']
    else:
        img_size = d_cfg['test_size']

    # build YOWO
    model = YOWOF(
        cfg=m_cfg,
        device=device,
        img_size=img_size,
        anchor_size=m_cfg['anchor_size'],
        len_clip=d_cfg['len_clip'],
        num_classes=num_classes,
        conf_thresh=m_cfg['conf_thresh'],
        nms_thresh=m_cfg['nms_thresh'],
        topk=args.topk,
        trainable=trainable
        )
            
    # keep training       
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
