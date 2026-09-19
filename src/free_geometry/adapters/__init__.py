"""Lazy model registry. Importing the protocol never imports a model package."""

from importlib import import_module

REGISTRY = {
    "da3": ("da3", "DA3Adapter"),
    "vggt": ("vggt", "VGGTAdapter"),
    "omega": ("vggt_omega", "VGGTOmegaAdapter"),
    "pi3": ("pi3", "Pi3Adapter"),
    "dvlt": ("dvlt", "DVLTAdapter"),
}


def get_adapter(config):
    if config.name not in REGISTRY:
        raise ValueError(f"unknown model: {config.name}")
    module, name = REGISTRY[config.name]
    return getattr(import_module("." + module, __name__), name)().configure(config)
