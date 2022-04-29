from .dilated_encoder import DilatedEncoder


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'dilated_encoder':
        neck = DilatedEncoder(in_dim, 
                              out_dim, 
                              expand_ratio=cfg['expand_ratio'], 
                              dilation_list=cfg['dilation_list'],
                              act_type=cfg['act_type'])

    return neck
