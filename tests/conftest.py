"""Shared fixtures for the rayfronts multi-robot tests.

Two environments have to run these:

* the **host**, which has numpy and pytest but no torch, no rclpy and no hydra.
  Only the pure tests run there:
  ``python3 -m pytest tests -q -m "not ros and not cuda and not torch"``
* the **robot container**, which has all of it (plus the compiled
  ``rayfronts_cpp``)::

      docker run --rm --network none --shm-size=1g \
        -v .../common/rayfronts:/root/AirStack/common/rayfronts \
        -v .../common/rayfronts_configs:/root/AirStack/common/rayfronts_configs:ro \
        -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST -e ROS_DOMAIN_ID=77 \
        -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        -e PYTHONPATH=/root/AirStack/common/rayfronts:/opt/rayfronts/rayfronts/csrc/build \
        <robot image> bash -lc 'source /opt/ros/jazzy/setup.bash \
          && cd /root/AirStack/common/rayfronts \
          && python3 -m pytest tests -q -m "not cuda" -p no:cacheprovider'

  ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` is NOT optional: sourcing the ROS setup
  puts ``launch_testing``/``launch_testing_ros`` on the entry-point path, and
  their pytest hooks are incompatible with the pytest in this image -- without
  it pytest dies with ``PluginValidationError`` before collecting anything.

  Add ``-m cuda`` on a GPU box for the CUDA-IPC test.

``rayfronts/__init__.py`` imports the heavy sub-packages, so on the host even
``import rayfronts.multi_robot_common`` would pull in torch. The pure modules
are therefore loaded straight off disk with :func:`load_standalone`, which
never executes the package ``__init__``.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "rayfronts"
CONFIGS = PKG / "configs"
REPO_CONFIGS = REPO.parent / "rayfronts_configs"


def pytest_configure(config):
  config.addinivalue_line("markers", "ros: needs rclpy and a working DDS")
  config.addinivalue_line("markers", "torch: needs torch (and the rayfronts "
                                     "package to be importable)")
  config.addinivalue_line("markers", "cuda: needs a real GPU")
  config.addinivalue_line("markers", "slow: takes more than a few seconds")


def load_standalone(module_name: str):
  """Import ``rayfronts/<module_name>.py`` WITHOUT running the package init.

  Only valid for modules that import nothing from ``rayfronts`` themselves --
  ``multi_robot_common`` and ``encoder_protocol``. That restriction is the
  point: it is what keeps their logic testable on a bare python.
  """
  path = PKG / f"{module_name}.py"
  key = f"_standalone_rayfronts_{module_name}"
  if key in sys.modules:
    return sys.modules[key]
  spec = importlib.util.spec_from_file_location(key, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[key] = mod
  spec.loader.exec_module(mod)
  return mod


def _have(mod_name: str) -> bool:
  try:
    return importlib.util.find_spec(mod_name) is not None
  except (ImportError, ValueError):
    return False


HAVE_TORCH = _have("torch")
HAVE_RCLPY = _have("rclpy")
HAVE_HYDRA = _have("hydra")


@pytest.fixture(scope="session")
def mrc():
  """rayfronts.multi_robot_common, loaded without the package init."""
  return load_standalone("multi_robot_common")


@pytest.fixture(scope="session")
def proto():
  """rayfronts.encoder_protocol, loaded without the package init."""
  return load_standalone("encoder_protocol")


@pytest.fixture(scope="session")
def pkg_root():
  return REPO


@pytest.fixture(scope="session")
def config_dirs():
  """(package configs, repo-side configs) as strings."""
  return str(CONFIGS), str(REPO_CONFIGS)


def ensure_pythonpath():
  """Make ``import rayfronts`` resolve to THIS checkout."""
  if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
  csrc = "/opt/rayfronts/rayfronts/csrc/build"
  if os.path.isdir(csrc) and csrc not in sys.path:
    sys.path.append(csrc)
