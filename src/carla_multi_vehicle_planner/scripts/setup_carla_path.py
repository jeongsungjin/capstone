import os
import sys

CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/ctrl/carla")
CARLA_EGG = os.path.join(
    CARLA_ROOT,
    "PythonAPI",
    "carla",
    "dist",
    "carla-0.9.16-py3.8-linux-x86_64.egg",
)
if os.path.exists(CARLA_EGG) and CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

AGENTS_ROOT = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if os.path.isdir(AGENTS_ROOT) and AGENTS_ROOT not in sys.path:
    sys.path.append(AGENTS_ROOT)
