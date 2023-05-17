

# DARMO 

Darts family models pre-trained

[![PyPI version](https://badge.fury.io/py/darmo.svg)](https://badge.fury.io/py/darmo)
![PyPI Release](https://github.com/jitdee-ai/darts-models/workflows/PyPI%20Release/badge.svg)
[![DOI](https://zenodo.org/badge/307382940.svg)](https://zenodo.org/badge/latestdoi/307382940)

## What's New

May 17, 2023
 - Add Vision GNN ImageNet-1k models, Thank from [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)

Feb 2, 2022
 - Add AutoFormerV2 ImageNet-1k models, Thank from [AutoFormerV2](https://github.com/silent-chen/AutoFormerV2-model-zoo)

Oct 6, 2021
 - Add PNASNet5 ImageNet-1k models, Thank from [PNASNet5](https://github.com/samyak0210/saliency/)

Sep 26, 2021
 - Add ResT ImageNet-1k models, Thank from [ResT](https://github.com/wofmanaf/ResT)

Sep 20, 2021
 - Add OFA-595 ImageNet-1k and ImageNet-21k, Thank from [21k-weight](https://github.com/Alibaba-MIIL/ImageNet21K) 
 - Add TResNet-M ImageNet-21k, Thank from [21k-weight](https://github.com/Alibaba-MIIL/ImageNet21K)

Sep 5, 2021
 - Add ResNet50 ImageNet-1k and ImageNet-21k, Thank from [21k-weight](https://github.com/Alibaba-MIIL/ImageNet21K)

Aug 17, 2021
 - Add EEEA-Net-C1 model

Aug 3, 2021
 - Add EEEA-Net-C2 model

July 20, 2021
 - Add test imagenet sctipt and result

April 5, 2021
 - Add NASNet model
 - Set params auxiliary

Oct 27, 2020
 - Add DARTSv2, PDART, RelativeNAS models
 
## Supported Models
    
 - [nasnet](https://arxiv.org/abs/1707.07012)
 - [dartsv2](https://github.com/quark0/darts)
 - [pdarts](https://github.com/chenxin061/pdarts)
 - [relative_nas](https://github.com/EMI-Group/RelativeNAS)
 - [eeea_c1, eeea_c2](https://github.com/chakkritte/EEEA-Net)
 - [resnet50_1k, resnet50_21k](https://arxiv.org/abs/1512.03385)
 - [ofa595_1k, ofa595_21k](https://github.com/mit-han-lab/once-for-all)
 - [tresnet_m_21k](https://github.com/Alibaba-MIIL/TResNet)
 - [rest_lite, rest_small, rest_base, rest_large](https://github.com/Alibaba-MIIL/TResNet)
 - [pnas5](https://github.com/samyak0210/saliency/)
 - [autoformerv2_tiny, autoformerv2_small, autoformerv2_base](https://github.com/silent-chen/AutoFormerV2-model-zoo)
 - [pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu](https://arxiv.org/abs/2206.00272)

## Install

The library can be installed with pip:

    pip install darmo

## Create models

    import darmo
    
    # just change -> "dartsv2", "pdarts", "relative_nas", "nasnet", "eeea_c2", "eeea_c1"
    model = darmo.create_model("dartsv2", num_classes=1000, pretrained=True, auxiliary=True)

## Supported Transfer learning
    # create model with ImageNet pretrained
	model = darmo.create_model("dartsv2", num_classes=1000, pretrained=True, auxiliary=True)
	
    # Reset classifier layer with add dropout before classifier layer
	model.reset_classifier(num_classes=100, dropout=0.2)

## Test model on ImageNet
    git clone https://github.com/jitdee-ai/darmo/
    cd darmo
    python test_imagenet.py --arch relative_nas --data [path of imagenet include val folder]

## Results of ImageNet

| Model | Top-1 Acc | Top-5 Acc | 
|--|--|--|
| dartsv2 | 73.59 | 91.40 |
| pdarts | 75.94 | 92.74 |
| eeea_c1 | 74.19 | 91.49 |
| eeea_c2 | 76.17| 92.66 |

## Citations this source code

    @software{chakkrit_termritthikun_2020_4139755,
    author       = {Chakkrit Termritthikun},
    title        = {jitdee-ai/darmo: pre-trained models for darts},
    month        = oct,
    year         = 2020,
    publisher    = {Zenodo},
    version      = {0.0.4},
    doi          = {10.5281/zenodo.4139755},
    url          = {https://doi.org/10.5281/zenodo.4139755}
    }
