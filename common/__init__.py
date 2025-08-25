"""Common utilities shared across training, inference, MQTT, and dashboard.

This package provides:
- model: CNN architecture for CIFAR-10–sized images
- data: datasets, transforms, and image encode/decode helpers
- inference: model load and prediction helpers
- mqtt: thin wrappers around paho-mqtt client setup and JSON publish
"""

from . import model as model
from . import data as data
from . import inference as inference
from . import mqtt as mqtt

__all__ = [
    "model",
    "data",
    "inference",
    "mqtt",
]

