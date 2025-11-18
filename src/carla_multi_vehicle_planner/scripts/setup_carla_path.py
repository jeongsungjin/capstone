import os
import sys

# Resolve CARLA Python path with priority:
# 1) CARLA_PYTHON_PATH (full path to carla build lib)
# 2) CARLA_ROOT (root dir, default ~/carla)
# 3) Fallback to ~/carla layout
_env_path = os.environ.get("CARLA_PYTHON_PATH")
if _env_path:
    CARLA_BUILD_PATH = _env_path
else:
    CARLA_ROOT = os.path.expanduser(os.environ.get("CARLA_ROOT", "~/carla"))
    CARLA_BUILD_PATH = os.path.join(
        CARLA_ROOT,
        "PythonAPI",
        "carla",
        "build",
        "lib.linux-x86_64-cpython-38",
    )
if os.path.exists(CARLA_BUILD_PATH) and CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

AGENTS_ROOT = None
try:
    # Use CARLA_ROOT if defined above; otherwise derive from build path
    if 'CARLA_ROOT' in globals():
        AGENTS_ROOT = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
    else:
        # CARLA_BUILD_PATH/.../carla/build/lib... → root is 4 levels up
        _bp = os.path.abspath(CARLA_BUILD_PATH)
        AGENTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(_bp)), "carla")
except Exception:
    AGENTS_ROOT = None
if AGENTS_ROOT and os.path.isdir(AGENTS_ROOT) and AGENTS_ROOT not in sys.path:
    sys.path.append(AGENTS_ROOT)
