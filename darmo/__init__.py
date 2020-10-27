from .version import __version__
from .factory import create_model

from .registry import *
from .network import NetworkImageNet
from .registry import register_model

from collections import namedtuple
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

url_cfgs = {
    'dartsv2': 'https://github.com/jitdee-ai/darts-models/releases/download/0.0.1/dartv2.pt',
    'pdarts' : 'https://github.com/jitdee-ai/darts-models/releases/download/0.0.1/pdarts.pt',
    'relative_nas' : 'https://github.com/jitdee-ai/darts-models/releases/download/0.0.1/relative_nas.pt',
}

def _remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def _set_config(_config={}, name=None, first_channels=48, layers=14, auxiliary=True, 
                        genotype=None, last_bn=False, pretrained=False, num_classes=1000):

    _config['name'] = name
    _config['first_channels'] = first_channels
    _config['layers'] = layers
    _config['auxiliary'] = auxiliary
    _config['genotype'] = genotype
    _config['last_bn'] = last_bn
    _config['pretrained'] = pretrained
    _config['num_classes'] = num_classes
    return _config

def _load_pre_trained(config):
    base_net = NetworkImageNet(
        config['first_channels'], 
        config['num_classes'], 
        config['layers'], 
        config['auxiliary'], 
        config['genotype'],
        config['last_bn'])

    if config['pretrained']:
        state_dict = model_zoo.load_url(url_cfgs[config['name']], progress=True, map_location='cpu')
        try:
            base_net.load_state_dict(state_dict, strict=True)
            print("loaded state dict")
        except RuntimeError:
            base_net.load_state_dict(_remove_module(state_dict), strict=True)
            print("loaded state dict")
    return base_net

@register_model
def dartsv2(pretrained=True, num_classes=1000):
    dartsv2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
    config = _set_config(_config={}, name= 'dartsv2', first_channels=48, layers=14, auxiliary=True, 
                        genotype=dartsv2, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)

@register_model
def pdarts(pretrained=True, num_classes=1000):

    pdarts = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

    config = _set_config(_config={}, name= 'pdarts', first_channels=48, layers=14, auxiliary=True, 
                        genotype=pdarts, last_bn=True, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)

@register_model
def relative_nas(pretrained=True, num_classes=1000):

    relative_nas = Genotype( normal=[ ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 0) ], normal_concat=range(2, 6), reduce=[ ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1) ], reduce_concat=range(2, 6) )
    
    config = _set_config(_config={}, name= 'relative_nas', first_channels=46, layers=14, auxiliary=True, 
                        genotype=relative_nas, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)