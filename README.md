
# DARMO 

Darts family models pre-trained

[![PyPI version](https://badge.fury.io/py/darmo.svg)](https://badge.fury.io/py/darmo)
![PyPI Release](https://github.com/jitdee-ai/darts-models/workflows/PyPI%20Release/badge.svg)

## What's New

Oct 27, 2020
 - Add DARTSv2, PDART, RelativeNAS models
 
## Supported Models

 - dartsv2
 - pdarts
 - relative_nas

## Install

The library can be installed with pip:

    pip install darmo

## Create models

    import darmo
    model = darmo.create_model("dartsv2", num_classes=1000, pretrained=True)

## Supported Transfer learning
    # create model with ImageNet pretrained
	model = darmo.create_model("dartsv2", num_classes=1000, pretrained=True)
    # Reset classifier layer with add dropout before classifier layer
	model.reset_classifier(num_classes=100, dropout=0.2)

