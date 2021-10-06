from .version import __version__
from .factory import create_model
from .registry import *
from .network import NetworkImageNet
from .registry import register_model
from .utils import _remove_module, _set_config

from collections import namedtuple
import torch.utils.model_zoo as model_zoo
import json
import pkg_resources

from .resnet import *
from .tresnet import *
from .nasnet import NASNetAMobile
from .nsga import *

from .models.rest import *
from .models.pnasnet5 import PNASNetwork

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

url_cfgs = {
    'dartsv2': 'https://github.com/jitdee-ai/darts-models/releases/download/0.0.1/dartv2.pt',
    'pdarts' : 'https://github.com/jitdee-ai/darts-models/releases/download/0.0.1/pdarts.pt',
    'relative_nas' : 'https://github.com/jitdee-ai/darts-models/releases/download/0.0.1/relative_nas.pt',
    'nasnet' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/nasnetamobile-7e03cead.pth',
    'eeea_c1' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/eeea_c1.pt',
    'eeea_c2' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/eeea-c2.pt',
    'pnas5' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/PNASNet-5_Large.pth',
}

def _load_pre_trained(config):
    if config['name'] == 'nasnet':
        base_net = NASNetAMobile()
    elif config['name'].startswith("eeea_"):
        config_file = pkg_resources.resource_filename(__name__, "config/"+config['name']+".config")
        subnet_file = pkg_resources.resource_filename(__name__, "config/"+config['name']+".subnet")
        config_subnet = json.load(open(config_file))
        subnet = json.load(open(subnet_file))
        base_net = NSGANetV2.build_from_config(config_subnet, depth=subnet['d'])
    else:
        if config['name'].startswith("pnas5"):
            base_net = PNASNetwork(
                config['first_channels'], 
                config['num_classes'], 
                config['layers'], 
                config['auxiliary'], 
                config['genotype'])
        else: 
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
def dartsv2(pretrained=True, num_classes=1000, auxiliary=False):
    dartsv2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
    config = _set_config(_config={}, name= 'dartsv2', first_channels=48, layers=14, auxiliary=auxiliary, 
                        genotype=dartsv2, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)

@register_model
def pdarts(pretrained=True, num_classes=1000, auxiliary=False):

    pdarts = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

    config = _set_config(_config={}, name= 'pdarts', first_channels=48, layers=14, auxiliary=auxiliary, 
                        genotype=pdarts, last_bn=True, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)

@register_model
def relative_nas(pretrained=True, num_classes=1000, auxiliary=False):

    relative_nas = Genotype( normal=[ ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 0) ], normal_concat=range(2, 6), reduce=[ ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1) ], reduce_concat=range(2, 6) )
    
    config = _set_config(_config={}, name= 'relative_nas', first_channels=46, layers=14, auxiliary=auxiliary, 
                        genotype=relative_nas, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)

@register_model
def nasnet(pretrained=True, num_classes=1000, auxiliary=True):

    config = _set_config(_config={}, name= 'nasnet', first_channels=46, layers=14, auxiliary=auxiliary, 
                        genotype=None, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _load_pre_trained(config)


@register_model
def pnas5(pretrained=True, num_classes=1000, auxiliary=False):

    pnas5 = Genotype( normal = [ ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_7x7', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ], normal_concat = [2, 3, 4, 5, 6], reduce = [ ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_7x7', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ], reduce_concat = [2, 3, 4, 5, 6], )
    config = _set_config(_config={}, name= 'pnas5', first_channels=216, layers=12, auxiliary=False, 
                        genotype=pnas5, last_bn=False, pretrained=pretrained, num_classes=1001)
    return _load_pre_trained(config)