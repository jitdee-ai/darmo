dependencies = ['torch']
from darmo.models import registry

globals().update(registry._model_entrypoints)