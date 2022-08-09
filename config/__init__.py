from .dataset_config import dataset_config
from .yowof_config import yowof_config


def build_model_config(args):
    print('==============================')
    print('Model Config: {} '.format(args.version.upper()))
    
    if args.version in ['yowof-r18', 'yowof-r50']:
        m_cfg = yowof_config[args.version]


    return m_cfg


def build_dataset_config(args):
    print('==============================')
    print('Dataset Config: {} '.format(args.dataset.upper()))
    
    d_cfg = dataset_config[args.dataset]

    return d_cfg
