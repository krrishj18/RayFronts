"""The single-robot path must be byte-for-byte what it was.

The existing deployment (`rayfronts.launch.xml` -> `python3 -m
rayfronts.mapping_server --config-name low_memory`) runs one process per robot
on the process-global default context. Adding the `context`/`domain_id`/
`node_name` kwargs must not change any of that: same node names, same spinner
thread names, same default-context behaviour, same `shutdown()` semantics.
"""

import inspect
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import HAVE_RCLPY, HAVE_TORCH, REPO, ensure_pythonpath

pytestmark = [
  pytest.mark.ros,
  pytest.mark.skipif(not (HAVE_RCLPY and HAVE_TORCH),
                     reason="needs rclpy and torch"),
]

if HAVE_RCLPY and HAVE_TORCH:
  ensure_pythonpath()
  import torch
  from rayfronts.datasets.ros import Ros2Subscriber
  from rayfronts.visualizers.ros import Ros2Vis
  from rayfronts.messaging_services.ros import Ros2MessagingService


def _run(snippet, timeout=120):
  """Run a snippet in a fresh interpreter (contexts do not survive teardown)."""
  env = dict(os.environ)
  env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
  return subprocess.run([sys.executable, "-c", textwrap.dedent(snippet)],
                        cwd=str(REPO), env=env, capture_output=True,
                        text=True, timeout=timeout)


# --------------------------------------------------------------------------- #
# Signatures: the new kwargs are additive and default to the old behaviour.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", ["Ros2Subscriber", "Ros2Vis",
                                 "Ros2MessagingService"])
def test_new_kwargs_are_optional_and_default_to_none(cls):
  obj = {"Ros2Subscriber": Ros2Subscriber, "Ros2Vis": Ros2Vis,
         "Ros2MessagingService": Ros2MessagingService}[cls]
  params = inspect.signature(obj.__init__).parameters
  for name in ("context", "domain_id", "node_name"):
    assert name in params, f"{cls} is missing the {name} kwarg"
    assert params[name].default is None, \
        f"{cls}.{name} must default to None so legacy callers are unaffected"


def test_existing_parameters_kept_their_order_and_defaults():
  """A positional caller (or a hydra config) must not shift under us."""
  p = list(inspect.signature(Ros2Subscriber.__init__).parameters)
  assert p[:12] == ["self", "rgb_topic", "pose_topic", "rgb_resolution",
                    "depth_resolution", "disparity_topic", "depth_topic",
                    "confidence_topic", "point_cloud_topic",
                    "intrinsics_topic", "intrinsics_file", "src_coord_system"]
  # The new kwargs are appended, not inserted.
  assert p[-3:] == ["context", "domain_id", "node_name"]

  v = list(inspect.signature(Ros2Vis.__init__).parameters)
  assert v[:8] == ["self", "intrinsics_3x3", "img_size", "base_point_size",
                   "global_heat_scale", "feat_compressor", "topic_prefix",
                   "reliability"]

  m = list(inspect.signature(Ros2MessagingService.__init__).parameters)
  assert m[:6] == ["self", "text_query_topic", "text_query_callback",
                   "query_publish_threshold", "topic_prefix", "frame_id"]


def test_sanitizer_still_lives_on_the_class():
  """raven_nav's label parsing depends on this exact behaviour."""
  svc = Ros2MessagingService.__new__(Ros2MessagingService)
  assert svc._sanitize_topic_name("fallen tree") == "fallen_tree"
  assert svc._sanitize_topic_name("a/b") == "a_b"
  assert svc._sanitize_topic_name("  x  ") == "x"
  assert svc._sanitize_topic_name("") == ""
  assert svc._query_topic_suffix(0, ["person"]) == "q0_person"
  assert svc._query_topic_suffix(3, None) == "q3"


def test_reserved_key_constants_are_unchanged():
  from rayfronts.messaging_services import ros as ros_ms
  assert ros_ms.KEY_VOXEL_SIMILARITY == "voxels_sim"
  assert ros_ms.KEY_RAY_SIMILARITY == "rays_sim"
  assert ros_ms.KEY_ALL_QUERIES == "all"


# --------------------------------------------------------------------------- #
# Runtime: no kwargs -> the global default context, the old node names.
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_legacy_objects_use_the_default_context_and_old_node_names():
  r = _run("""
    import torch, rclpy
    from rayfronts.visualizers.ros import Ros2Vis
    from rayfronts.messaging_services.ros import Ros2MessagingService

    intr = torch.eye(3)
    vis = Ros2Vis(intrinsics_3x3=intr, base_point_size=0.1)
    svc = Ros2MessagingService(text_query_topic="rayfronts/msg_serv/new_text_query")

    assert vis._rosnode.get_name() == "rayfronts_vis", vis._rosnode.get_name()
    assert svc._rosnode.get_name() == "rayfronts_messaging_service"
    # Both must be on the ONE process-global default context.
    assert vis._context is None and svc._context is None
    assert vis._owns_context is False and svc._owns_context is False
    assert vis._rosnode.context is rclpy.get_default_context()
    assert svc._rosnode.context is vis._rosnode.context
    names = {t.name for t in
             __import__("threading").enumerate() if t.name}
    assert "rayfronts_vis_spinner" in names, names
    assert "rayfronts_messaging_service_spinner" in names, names
    print("LEGACY_OK")
  """)
  assert "LEGACY_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.slow
def test_legacy_shutdown_still_tears_down_the_default_context():
  r = _run("""
    import torch, rclpy
    from rayfronts.visualizers.ros import Ros2Vis
    vis = Ros2Vis(intrinsics_3x3=torch.eye(3), base_point_size=0.1)
    ctx = rclpy.get_default_context()
    assert ctx.ok()
    vis.shutdown()
    assert not ctx.ok(), "legacy shutdown() must still try_shutdown() the context"
    print("LEGACY_SHUTDOWN_OK")
  """)
  assert "LEGACY_SHUTDOWN_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.slow
def test_a_private_context_does_not_touch_the_default_one():
  """The bug the multi-robot server exists to avoid."""
  r = _run("""
    import torch, rclpy
    from rayfronts.visualizers.ros import Ros2Vis
    from rayfronts import ros_context

    legacy = Ros2Vis(intrinsics_3x3=torch.eye(3), base_point_size=0.1)
    scoped = Ros2Vis(intrinsics_3x3=torch.eye(3), base_point_size=0.1,
                     domain_id=89, node_name="rayfronts_vis_robot_9",
                     topic_prefix="/robot_9/rayfronts")

    assert scoped._rosnode.get_name() == "rayfronts_vis_robot_9"
    assert scoped._context is not None and scoped._owns_context
    assert scoped._rosnode.context is not rclpy.get_default_context()
    assert ros_context.refcount(89) == 1

    scoped.shutdown()
    assert ros_context.refcount(89) == 0
    # The legacy object is untouched: this is the regression that used to
    # happen when every rayfronts object shared one context.
    assert rclpy.get_default_context().ok()
    assert legacy._rosnode.context.ok()
    print("ISOLATION_OK")
  """)
  assert "ISOLATION_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.slow
def test_two_components_on_one_domain_share_one_context():
  r = _run("""
    import torch
    from rayfronts.visualizers.ros import Ros2Vis
    from rayfronts.messaging_services.ros import Ros2MessagingService
    from rayfronts import ros_context

    vis = Ros2Vis(intrinsics_3x3=torch.eye(3), base_point_size=0.1,
                  domain_id=88, node_name="v88")
    svc = Ros2MessagingService(text_query_topic="/x/new_text_query",
                               domain_id=88, node_name="m88")
    assert vis._context is svc._context, "same domain must share a context"
    assert ros_context.refcount(88) == 2
    vis.shutdown()
    assert ros_context.refcount(88) == 1
    assert svc._context.ok(), "one component's shutdown killed the other's"
    svc.shutdown()
    assert ros_context.refcount(88) == 0
    print("SHARED_CTX_OK")
  """)
  assert "SHARED_CTX_OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.slow
def test_mapping_server_module_still_imports_and_composes():
  """The legacy entry point must still be startable."""
  r = _run("""
    import rayfronts.mapping_server as ms
    assert hasattr(ms, "MappingServer") and hasattr(ms, "main")
    import rayfronts.multi_robot_mapping_server as mms
    assert issubclass(mms.MultiRobotMappingServer, ms.MappingServer)
    # The subclass must not have broken the base's public surface.
    for name in ("add_queries", "clear_queries", "run_queries", "run",
                 "shutdown"):
      assert hasattr(mms.MultiRobotMappingServer, name), name
    print("IMPORT_OK")
  """)
  assert "IMPORT_OK" in r.stdout, r.stdout + r.stderr
