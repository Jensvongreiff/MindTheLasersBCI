import builtins
import importlib
import sys
import types
import unittest
from unittest.mock import patch


class InputControllerImportTests(unittest.TestCase):
    def test_import_does_not_require_sklearn_metrics_at_module_load(self):
        sys.modules.pop("mind_the_lasers.src.game.input_controller", None)

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sklearn.metrics" or name.startswith("sklearn.metrics."):
                raise ImportError("blocked for import test")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            module = importlib.import_module("mind_the_lasers.src.game.input_controller")

        self.assertTrue(hasattr(module, "KeyboardController"))
        self.assertTrue(hasattr(module, "BCIController"))


if __name__ == "__main__":
    unittest.main()
