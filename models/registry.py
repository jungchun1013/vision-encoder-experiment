from wrappers.encoder import BaseEncoder

_REGISTRY: dict[str, type[BaseEncoder]] = {}


def register(name: str):
    """Decorator to register an encoder class under a string name."""
    def decorator(cls: type[BaseEncoder]):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_encoder(name: str, **kwargs) -> BaseEncoder:
    """Instantiate an encoder by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_encoders() -> list[str]:
    """Return sorted list of registered encoder names."""
    return sorted(_REGISTRY)
