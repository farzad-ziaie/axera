"""
Plugin and hook system for Axera.

Allows registering custom pre-processing and post-processing functions
without forking the package.  Hooks are called in registration order.

Usage
-----
>>> from axera.hooks import HookRegistry
>>> registry = HookRegistry()
>>> @registry.register("pre_predict")
... def clip_inputs(x):
...     return x.clip(-10, 10)
>>> x_processed = registry.run("pre_predict", x_raw)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ── Hook types ────────────────────────────────────────────────────────────────

HookFn = Callable[..., Any]

VALID_HOOKS = frozenset({
    "pre_fit",          # called before training starts  (X, y) -> (X, y)
    "post_fit",         # called after training          (model) -> model
    "pre_predict",      # called before prediction       (X,) -> (X,)
    "post_predict",     # called after prediction        (pred,) -> (pred,)
    "on_epoch_start",   # called at epoch start          (epoch, logs) -> None
    "on_epoch_end",     # called at epoch end            (epoch, logs) -> None
    "on_loss",          # called after loss computation  (loss,) -> loss
})


class HookRegistry:
    """
    Central registry for Axera processing hooks.

    Each hook slot holds an ordered list of callables.  Calling
    ``run(hook_name, *args)`` pipes the output of each function to the
    next (for transform hooks) or calls them in order (for event hooks).
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFn]] = defaultdict(list)

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, hook: str, fn: HookFn | None = None): # noqa: ANN201
        """
        Register a hook function.

        Can be used as a decorator or called directly.

        Parameters
        ----------
        hook : str
            Hook name (must be in ``VALID_HOOKS``).
        fn : callable, optional
            The function to register.  If ``None``, returns a decorator.

        Examples
        --------
        As decorator::

            @registry.register("pre_predict")
            def my_transform(x):
                return x / x.std()

        Direct call::

            registry.register("pre_predict", my_transform)
        """
        if hook not in VALID_HOOKS:
            raise ValueError(f"Unknown hook: {hook!r}. Valid: {sorted(VALID_HOOKS)}")

        def decorator(f: HookFn) -> HookFn:
            self._hooks[hook].append(f)
            logger.debug("Registered hook '%s': %s", hook, f.__name__)
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def unregister(self, hook: str, fn: HookFn) -> None:
        """Remove a specific function from a hook slot."""
        try:
            self._hooks[hook].remove(fn)
        except ValueError:
            logger.warning("Hook '%s' did not contain %s", hook, fn.__name__)

    def clear(self, hook: str | None = None) -> None:
        """Clear all hooks, or only those for a specific ``hook`` name."""
        if hook is None:
            self._hooks.clear()
        else:
            self._hooks[hook].clear()

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, hook: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute all registered functions for ``hook``.

        For *transform* hooks (pre_fit, post_fit, pre_predict,
        post_predict, on_loss) the output of each function is piped as
        the first argument to the next.

        For *event* hooks (on_epoch_start, on_epoch_end) all functions
        are called with the same arguments and return values are ignored.
        """
        fns = self._hooks.get(hook, [])
        if not fns:
            return args[0] if len(args) == 1 else args

        event_hooks = {"on_epoch_start", "on_epoch_end"}
        if hook in event_hooks:
            for fn in fns:
                fn(*args, **kwargs)
            return None

        # Transform pipe
        result = args[0] if len(args) == 1 else args
        for fn in fns:
            if isinstance(result, tuple):
                result = fn(*result)
            else:
                result = fn(result)
        return result

    def __repr__(self) -> str:
        registered = {k: [f.__name__ for f in v] for k, v in self._hooks.items() if v}
        return f"HookRegistry({registered})"


# ── Global default registry ───────────────────────────────────────────────────

_global_registry: HookRegistry = HookRegistry()


def register(hook: str, fn: HookFn | None = None): # noqa: ANN201
    """Register a hook in the global default registry."""
    return _global_registry.register(hook, fn)


def get_global_registry() -> HookRegistry:
    """Return the global default hook registry."""
    return _global_registry


__all__ = [
    "HookRegistry",
    "HookFn",
    "VALID_HOOKS",
    "register",
    "get_global_registry",
]
