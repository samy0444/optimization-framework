"""Plugin registry: discover domains, optimizers and executors by name.

Lets the CLI / API say `--domain perovskite --optimizer bayesian` without
importing concrete classes. Plugins self-register via the @register_* decorators.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from sdl.core.domain import Domain
from sdl.core.executor import Executor
from sdl.core.optimizer import Optimizer

_DOMAINS: Dict[str, Type[Domain]] = {}
_OPTIMIZERS: Dict[str, Type[Optimizer]] = {}
_EXECUTORS: Dict[str, Type[Executor]] = {}


def register_domain(name: str) -> Callable[[Type[Domain]], Type[Domain]]:
    def deco(cls: Type[Domain]) -> Type[Domain]:
        cls.name = name
        _DOMAINS[name] = cls
        return cls
    return deco


def register_optimizer(name: str) -> Callable[[Type[Optimizer]], Type[Optimizer]]:
    def deco(cls: Type[Optimizer]) -> Type[Optimizer]:
        cls.name = name
        _OPTIMIZERS[name] = cls
        return cls
    return deco


def register_executor(name: str) -> Callable[[Type[Executor]], Type[Executor]]:
    def deco(cls: Type[Executor]) -> Type[Executor]:
        cls.name = name
        _EXECUTORS[name] = cls
        return cls
    return deco


def get_domain(name: str) -> Type[Domain]:
    _ensure_builtins_loaded()
    if name not in _DOMAINS:
        raise KeyError(f"Unknown domain '{name}'. Available: {sorted(_DOMAINS)}")
    return _DOMAINS[name]


def get_optimizer(name: str) -> Type[Optimizer]:
    _ensure_builtins_loaded()
    if name not in _OPTIMIZERS:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}")
    return _OPTIMIZERS[name]


def get_executor(name: str) -> Type[Executor]:
    _ensure_builtins_loaded()
    if name not in _EXECUTORS:
        raise KeyError(f"Unknown executor '{name}'. Available: {sorted(_EXECUTORS)}")
    return _EXECUTORS[name]


def available() -> Dict[str, list]:
    _ensure_builtins_loaded()
    return {
        "domains": sorted(_DOMAINS),
        "optimizers": sorted(_OPTIMIZERS),
        "executors": sorted(_EXECUTORS),
    }


_LOADED = False


def _ensure_builtins_loaded() -> None:
    """Import built-in plugins once so their decorators run."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import importlib

    for mod in (
        "sdl.optimizers.random_search",
        "sdl.optimizers.bayesian",
        "sdl.optimizers.physics_bo",
        "sdl.optimizers.rl",
        "sdl.optimizers.hybrid",
        "sdl.executors.simulated",
        "sdl.executors.tabular",
        "sdl.domains.reaction",
        "sdl.domains.branin",
        "sdl.domains.perovskite",
        "sdl.domains.tabular",
    ):
        try:
            importlib.import_module(mod)
        except Exception:
            # A plugin with missing optional deps must not break the registry.
            pass
