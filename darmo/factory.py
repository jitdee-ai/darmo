from .registry import is_model, is_model_in_modules, model_entrypoint

def create_model(
        model_name,
        pretrained=True,
        num_classes=1000,
        auxiliary=True,
        **kwargs):
    """Create a model
    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
    """
    model_args = dict(pretrained=pretrained, num_classes=num_classes, auxiliary=auxiliary)

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**model_args, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)
    
    return model
