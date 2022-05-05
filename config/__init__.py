from .dataset_config import dataset_config
from .model_config import model_config


def build_model_config(args):
    print('==============================')
    print('Build Model: {} '.format(args.version.upper()))
    
    m_cfg = model_config[args.version]

    return m_cfg


def build_dataset_config(args):
    print('==============================')
    print('Build Dataset: {} '.format(args.version.upper()))
    
    d_cfg = dataset_config[args.dataset]

    return d_cfg
