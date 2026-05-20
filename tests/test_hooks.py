"""Tests for axera.hooks — plugin/hook system."""

from __future__ import annotations

import numpy as np
import pytest
from axera.hooks import VALID_HOOKS, HookRegistry


class TestHookRegistry:
    def test_register_and_run_transform(self):
        reg = HookRegistry()

        @reg.register("pre_predict")
        def scale(x):
            return x * 2

        arr = np.array([1.0, 2.0, 3.0])
        out = reg.run("pre_predict", arr)
        np.testing.assert_allclose(out, arr * 2)

    def test_chain_transforms(self):
        reg = HookRegistry()
        reg.register("on_loss", lambda x: x + 1)
        reg.register("on_loss", lambda x: x * 2)
        out = reg.run("on_loss", 5)
        assert out == (5 + 1) * 2  # first +1, then *2

    def test_event_hooks_called_in_order(self):
        reg = HookRegistry()
        log = []
        reg.register("on_epoch_start", lambda e, h: log.append(("A", e)))
        reg.register("on_epoch_start", lambda e, h: log.append(("B", e)))
        reg.run("on_epoch_start", 1, {})
        assert log == [("A", 1), ("B", 1)]

    def test_invalid_hook_raises(self):
        reg = HookRegistry()
        with pytest.raises(ValueError):
            reg.register("nonexistent_hook")

    def test_unregister(self):
        reg = HookRegistry()
        fn = lambda x: x + 99
        reg.register("pre_predict", fn)
        reg.unregister("pre_predict", fn)
        out = reg.run("pre_predict", 1.0)
        assert out == 1.0   # fn was removed, pass-through

    def test_clear_all(self):
        reg = HookRegistry()
        reg.register("pre_predict", lambda x: x + 1)
        reg.register("post_predict", lambda x: x + 2)
        reg.clear()
        assert reg.run("pre_predict", 0.0) == 0.0

    def test_clear_single_hook(self):
        reg = HookRegistry()
        reg.register("pre_predict", lambda x: x + 1)
        reg.register("post_predict", lambda x: x + 2)
        reg.clear("pre_predict")
        assert reg.run("pre_predict", 0.0) == 0.0
        assert reg.run("post_predict", 0.0) == 2.0

    def test_no_hooks_passthrough(self):
        reg = HookRegistry()
        arr = np.array([1, 2, 3])
        out = reg.run("pre_predict", arr)
        np.testing.assert_array_equal(out, arr)

    def test_all_valid_hook_names(self):
        """Verify all VALID_HOOKS can be registered without error."""
        reg = HookRegistry()
        for hook in VALID_HOOKS:
            if hook.startswith("on_epoch"):
                reg.register(hook, lambda *a: None)
            else:
                reg.register(hook, lambda x: x)
