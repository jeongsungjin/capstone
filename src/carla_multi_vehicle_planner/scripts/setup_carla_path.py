import os
import sys

CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/jamie/carla")
CARLA_BUILD_PATH = os.path.join(
    CARLA_ROOT,
    "PythonAPI",
    "carla",
    "build",
    "lib.linux-x86_64-cpython-38",
)
if os.path.exists(CARLA_BUILD_PATH) and CARLA_BUILD_PATH not in sys.path:
    sys.path.insert(0, CARLA_BUILD_PATH)

AGENTS_ROOT = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if os.path.isdir(AGENTS_ROOT) and AGENTS_ROOT not in sys.path:
    sys.path.append(AGENTS_ROOT)
