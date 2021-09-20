import torch
from torch.autograd import Variable
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

def _load(config, base_net, url_cfgs, use_dict=False):
    if config['pretrained']:
        state_dict = model_zoo.load_url(url_cfgs[config['name']], progress=True, map_location='cpu')
        if use_dict:
            state_dict  = state_dict['state_dict']
        try:
            base_net.load_state_dict(state_dict, strict=True)
            print("loaded state dict")
        except RuntimeError:
            base_net.load_state_dict(_remove_module(state_dict), strict=True)
            print("loaded state dict")

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

def _remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x
