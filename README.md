
# DARMO 

Darts family models pre-trained

[![PyPI version](https://badge.fury.io/py/darmo.svg)](https://badge.fury.io/py/darmo)
![PyPI Release](https://github.com/jitdee-ai/darts-models/workflows/PyPI%20Release/badge.svg)
[![DOI](https://zenodo.org/badge/307382940.svg)](https://zenodo.org/badge/latestdoi/307382940)

## What's New

Oct 27, 2020
 - Add DARTSv2, PDART, RelativeNAS models
 
## Supported Models

 - [dartsv2](https://github.com/quark0/darts)
 - [pdarts](https://github.com/chenxin061/pdarts)
 - [relative_nas](https://github.com/EMI-Group/RelativeNAS)

## Install

The library can be installed with pip:

    pip install darmo

## Create models

    import darmo
    
    # just change -> "dartsv2", "pdarts", "relative_nas"
    model = darmo.create_model("dartsv2", num_classes=1000, pretrained=True)

## Supported Transfer learning
    # create model with ImageNet pretrained
	model = darmo.create_model("dartsv2", num_classes=1000, pretrained=True)
	
    # Reset classifier layer with add dropout before classifier layer
	model.reset_classifier(num_classes=100, dropout=0.2)

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
