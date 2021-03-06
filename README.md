

# DARMO 

Darts family models pre-trained

[![PyPI version](https://badge.fury.io/py/darmo.svg)](https://badge.fury.io/py/darmo)
![PyPI Release](https://github.com/jitdee-ai/darts-models/workflows/PyPI%20Release/badge.svg)
[![DOI](https://zenodo.org/badge/307382940.svg)](https://zenodo.org/badge/latestdoi/307382940)

## What's New

July 20, 2021
 - Add test imagenet sctipt and result

April 5, 2021
 - Add NASNet models
 - Set params auxiliary

Oct 27, 2020
 - Add DARTSv2, PDART, RelativeNAS models
 
## Supported Models
    
 - [nasnet](https://arxiv.org/abs/1707.07012)
 - [dartsv2](https://github.com/quark0/darts)
 - [pdarts](https://github.com/chenxin061/pdarts)
 - [relative_nas](https://github.com/EMI-Group/RelativeNAS)

## Install

The library can be installed with pip:

    pip install darmo

## Create models

    import darmo
    
    # just change -> "dartsv2", "pdarts", "relative_nas", "nasnet"
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
| relative_nas |  |  |
| nasnet |  |  |

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
