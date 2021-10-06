from timm.models.layers import drop_path
from ofa.imagenet_codebase.modules.layers import *
from ofa.layers import set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, LinearLayer
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock
import torch
import torch.nn as nn
import torch

from .registry import register_model
import json
import pkg_resources
from .utils import _set_config, _load

url_cfgs = {
    'eeea_c1' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/eeea_c1.pt',
    'eeea_c2' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/eeea-c2.pt',
    'ofa595_1k' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/ofa595_1k_b1a18d5.pt',
    'ofa595_21k' : 'https://github.com/jitdee-ai/darmo/releases/download/0.0.1/ofa595_21k_ec9428e.pt',
}

def _create_network(config):
    if config['name'].startswith("ofa595"):
        config_file = pkg_resources.resource_filename(__name__, "config/ofa595.config")
    else:
        config_file = pkg_resources.resource_filename(__name__, "config/"+config['name']+".config")
    config_subnet = json.load(open(config_file))

    if config['name'] == "ofa595_21k":
        config_subnet['classifier']['out_features'] = config['num_classes']

    base_net = NSGANetV2.build_from_config(config_subnet, depth=None)

    if config['name'] == "ofa595_21k":
        base_net.classifier = None

    _load(config, base_net, url_cfgs)

    #if config['name'] == "ofa595_21k":
    if config['name'].startswith("ofa595_"):
        NSGANetV2.reset_classifier(base_net, 1536, config['num_classes'], dropout_rate=0.0)
        base_net.num_features = 1536
    return base_net

@register_model
def ofa595_1k(pretrained=True, num_classes=1000, auxiliary=True):

    config = _set_config(_config={}, name= 'ofa595_1k', first_channels=None, layers=None, auxiliary=auxiliary, 
                        genotype=None, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _create_network(config)

@register_model
def ofa595_21k(pretrained=True, num_classes=11221, auxiliary=True):

    config = _set_config(_config={}, name= 'ofa595_21k', first_channels=None, layers=None, auxiliary=auxiliary, 
                        genotype=None, last_bn=False, pretrained=pretrained, num_classes=11221)

    return _create_network(config)

@register_model
def eeea_c1(pretrained=True, num_classes=1000, auxiliary=True):

    config = _set_config(_config={}, name= 'eeea_c1', first_channels=46, layers=14, auxiliary=auxiliary, 
                        genotype=None, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _create_network(config)

@register_model
def eeea_c2(pretrained=True, num_classes=1000, auxiliary=True):

    config = _set_config(_config={}, name= 'eeea_c2', first_channels=46, layers=14, auxiliary=auxiliary, 
                        genotype=None, last_bn=False, pretrained=pretrained, num_classes=num_classes)

    return _create_network(config)

class MobileInvertedResidualBlock(MyModule):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/master/ofa/
    imagenet_codebase/networks/proxyless_nets.py to include drop path in training

    """
    def __init__(self, mobile_inverted_conv, shortcut, drop_connect_rate=0.0):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            # res = self.mobile_inverted_conv(x) + self.shortcut(x)
            res = self.mobile_inverted_conv(x)

            if self.drop_connect_rate > 0.:
                res = drop_path(res, drop_prob=self.drop_connect_rate, training=self.training)

            res += self.shortcut(x)

        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(
            mobile_inverted_conv, shortcut, drop_connect_rate=config['drop_connect_rate'])

class MyNetwork(MyModule):

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError
    
    """ implemented methods """
    
    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier, depth):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.depth = depth
        self.channels = []
        self.num_features = 1280

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    def reset_classifier(self, num_classes, dropout=0.0):
        self.num_classes = num_classes
        self.drop_rate = dropout

        del self.classifier

        if self.num_classes:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
        else:
            self.classifier = None

class NSGANetV2(MobileNetV3):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/master/ofa/
    imagenet_codebase/networks/mobilenet_v3.py to include drop path in training
    and option to reset classification layer
    """
    @staticmethod
    def build_from_config(config, drop_connect_rate=0.0, depth=None):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_idx, block_config in enumerate(config['blocks']):
            block_config['drop_connect_rate'] = drop_connect_rate * block_idx / len(config['blocks'])
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier, depth)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    # @staticmethod
    # def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
    #     # first conv layer
    #     first_conv = ConvLayer(
    #         3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
    #     )
    #     # build mobile blocks
    #     feature_dim = input_channel
    #     blocks = []
    #     for stage_id, block_config_list in cfg.items():
    #         for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
    #             mb_conv = MBInvertedConvLayer(
    #                 feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
    #             )
    #             if stride == 1 and out_channel == feature_dim:
    #                 shortcut = IdentityLayer(out_channel, out_channel)
    #             else:
    #                 shortcut = None
    #             blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
    #             feature_dim = out_channel
    #     # final expand layer
    #     final_expand_layer = ConvLayer(
    #         feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
    #     )
    #     feature_dim = feature_dim * 6
    #     # feature mix layer
    #     feature_mix_layer = ConvLayer(
    #         feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
    #     )
    #     # classifier
    #     classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

    #     return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def reset_classifier(model, last_channel, n_classes, dropout_rate=0.0):
        model.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
