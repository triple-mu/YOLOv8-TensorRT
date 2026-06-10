"""Load the custom TensorRT plugin library (libyolov8_plugins.so) so its creators
register before any engine build or deserialize. Call load_plugin_lib() ahead of
trt.init_libnvinfer_plugins(). The path comes from the argument or the
YOLOV8_PLUGIN_LIB env var; if neither is set this is a no-op (stock engines are
unaffected)."""

import ctypes
import os

_loaded: set[str] = set()


def load_plugin_lib(path: str | None = None) -> None:
    path = path or os.environ.get("YOLOV8_PLUGIN_LIB")
    if not path or path in _loaded:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"plugin library not found: {path}")
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    _loaded.add(path)
