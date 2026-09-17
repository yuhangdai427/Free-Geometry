from .base import BaseAdapter  # noqa: F401


def get_adapter(model_key: str, **kw) -> BaseAdapter:
    if model_key == "omega":
        from .vggt_omega import VGGTOmegaAdapter
        return VGGTOmegaAdapter(**kw)
    if model_key == "pi3":
        from .pi3 import Pi3Adapter
        return Pi3Adapter(**kw)
    if model_key == "dvlt":
        from .dvlt import DVLTAdapter
        return DVLTAdapter(**kw)
    raise ValueError(f"unknown model {model_key}")
