"""MultiRobotRos2Subscriber against two fake robots on two ROS domains."""

import time

import numpy as np
import pytest

from conftest import HAVE_RCLPY, HAVE_TORCH, ensure_pythonpath

pytestmark = [
  pytest.mark.ros,
  pytest.mark.skipif(not (HAVE_RCLPY and HAVE_TORCH),
                     reason="needs rclpy and torch"),
]

if HAVE_RCLPY and HAVE_TORCH:
  ensure_pythonpath()
  import torch
  import ros_helpers as rh
  from rayfronts import ros_context
  from rayfronts import multi_robot_common as mrc
  from rayfronts.datasets.multi_ros import MultiRobotRos2Subscriber

D1, D2 = 90, 91


def _drive(robots, poses, n=40, t0=100.0, dt=0.1):
  """Publish n synchronised frames per robot."""
  for i in range(n):
    t = t0 + i * dt
    for r, xyz in zip(robots, poses):
      r.publish_frame(t, xyz)
    time.sleep(0.01)


@pytest.fixture(autouse=True)
def no_leaked_contexts():
  """Every test must leave the private-context registry exactly as it found it.

  A dataset that raises half way through construction used to keep the
  contexts of the robots it had already built, which is invisible until some
  later component wonders why its domain never shuts down.
  """
  watched = (D1, D2, 92, 93)
  before = {d: ros_context.refcount(d) for d in watched}
  yield
  after = {d: ros_context.refcount(d) for d in watched}
  assert after == before, f"rclpy context leak: {before} -> {after}"


@pytest.fixture
def robots():
  a = rh.FakeRobot(1, D1)
  b = rh.FakeRobot(2, D2)
  try:
    yield a, b
  finally:
    a.shutdown()
    b.shutdown()


def _make(robots, **kwargs):
  cfg = dict(robot_ids=[1, 2], domain_ids=[D1, D2],
             rgb_resolution=[16, 16], depth_resolution=[16, 16],
             frame_skip=0, intrinsics_timeout_s=30.0, sync_slop_s=0.05)
  cfg.update(kwargs)

  import threading
  stop = threading.Event()

  def pump():
    while not stop.is_set():
      for r in robots:
        r.info_pub.publish(r.camera_info())
      time.sleep(0.1)

  t = threading.Thread(target=pump, daemon=True)
  t.start()
  try:
    ds = MultiRobotRos2Subscriber(**cfg)
  finally:
    stop.set()
    t.join(timeout=5)
  return ds


# --------------------------------------------------------------------------- #

def test_intrinsics_come_from_camera_info(robots):
  ds = _make(robots)
  try:
    assert ds.intrinsics_3x3 is not None
    np.testing.assert_allclose(ds.intrinsics_3x3.numpy(),
                               [[8.0, 0, 8.0], [0, 8.0, 8.0], [0, 0, 1]])
    assert ds.rgb_h == ds.rgb_w == 16
    assert ds.original_h == ds.original_w == 16
  finally:
    ds.shutdown()


def test_topic_templates_expand_per_robot(robots):
  ds = _make(robots)
  try:
    t1 = ds.topics_for(1)
    t2 = ds.topics_for(2)
    assert t1["rgb"] == "/robot_1/sensors/front_stereo/left/image_rect"
    assert t2["rgb"] == "/robot_2/sensors/front_stereo/left/image_rect"
    assert t1["pose"] == "/robot_1/odometry_conversion/odometry"
    assert t2["navsat"] == \
        "/robot_2/interface/mavros/global_position/global"
    assert ds.domain_of(1) == D1 and ds.domain_of(2) == D2
  finally:
    ds.shutdown()


def test_round_robin_order_is_strict_when_both_queues_are_full(robots):
  """The scheduling rule itself, with no ROS timing in the way.

  Pre-load both per-robot queues, then drain them through the same
  round-robin sweep __iter__ uses: no robot may be served twice while the
  other has a frame waiting.
  """
  ds = _make(robots, anchor_mode="none")
  try:
    for i in range(4):
      for rid in (1, 2):
        ds.streams[rid].msgs.put({"marker": (rid, i)})
    order = []
    while True:
      item = ds._next_msgs()
      if item is None:
        break
      stream, msgs = item
      order.append(stream.robot_id)
    assert len(order) == 8, order
    assert all(order[i] != order[i + 1] for i in range(len(order) - 1)), order
  finally:
    ds.shutdown()


def test_round_robin_skips_an_empty_queue_instead_of_blocking(robots):
  """One silent robot must not stall the map for the other."""
  ds = _make(robots, anchor_mode="none")
  try:
    for i in range(3):
      ds.streams[1].msgs.put({"marker": i})
    order = []
    while True:
      item = ds._next_msgs()
      if item is None:
        break
      order.append(item[0].robot_id)
    assert order == [1, 1, 1], order
  finally:
    ds.shutdown()


@pytest.mark.slow
def test_both_robots_are_served_over_a_live_stream(robots):
  """End to end: neither robot starves while the other flies."""
  ds = _make(robots, anchor_mode="none")
  try:
    import threading
    stop = threading.Event()

    def pub():
      i = 0
      while not stop.is_set():
        t = 100.0 + i * 0.1
        for r, xyz in zip(robots, [(1, 2, 3), (4, 5, 6)]):
          r.publish_frame(t, xyz)
        i += 1
        time.sleep(0.02)

    th = threading.Thread(target=pub, daemon=True)
    th.start()
    seen = []
    it = iter(ds)
    deadline = time.time() + 60
    while len(seen) < 20 and time.time() < deadline:
      seen.append(next(it)["robot_id"])
    stop.set()
    th.join(timeout=5)

    assert len(seen) >= 20, f"only got {seen}"
    # Exact alternation is not guaranteed (a queue can be momentarily empty and
    # the sweep must not block), but neither robot may dominate.
    for rid in (1, 2):
      assert seen.count(rid) >= len(seen) // 3, seen
  finally:
    ds.shutdown()


@pytest.mark.slow
def test_static_anchoring_shifts_the_pose_into_the_world_frame(robots):
  """robot_2 spawned at (100, 50): its local (4,5) must land at world (104,55)."""
  ds = _make(robots, anchor_mode="static",
             robot_offsets_xy={1: [0.0, 0.0], 2: [100.0, 50.0]})
  try:
    np.testing.assert_allclose(ds.boot_enu(2), [100.0, 50.0, 0.0])
    assert ds.is_anchored(1) and ds.is_anchored(2)

    import threading
    stop = threading.Event()

    def pub():
      i = 0
      while not stop.is_set():
        t = 200.0 + i * 0.1
        robots[0].publish_frame(t, (1.0, 2.0, 3.0))
        robots[1].publish_frame(t, (4.0, 5.0, 6.0))
        i += 1
        time.sleep(0.02)

    th = threading.Thread(target=pub, daemon=True)
    th.start()
    got = dict()
    it = iter(ds)
    deadline = time.time() + 40
    while len(got) < 2 and time.time() < deadline:
      f = next(it)
      got.setdefault(f["robot_id"], f)
    stop.set()
    th.join(timeout=5)
    assert set(got) == {1, 2}

    # flu -> rdf: (x, y, z)_flu -> (-y, -z, x)_rdf
    p1 = got[1]["pose_4x4"][:3, 3].numpy()
    np.testing.assert_allclose(p1, [-2.0, -3.0, 1.0], atol=1e-4)
    p2 = got[2]["pose_4x4"][:3, 3].numpy()
    # world FLU = (4+100, 5+50, 6) = (104, 55, 6) -> rdf (-55, -6, 104)
    np.testing.assert_allclose(p2, [-55.0, -6.0, 104.0], atol=1e-3)

    # ...and that is exactly the documented shift applied to the unshifted pose.
    np.testing.assert_allclose(p2 - np.array([-5.0, -6.0, 4.0]),
                               mrc.local_to_world_shift([100.0, 50.0, 0.0]),
                               atol=1e-3)
    assert got[1]["robot_name"] == "robot_1"
    assert got[2]["robot_name"] == "robot_2"
  finally:
    ds.shutdown()


def test_static_anchoring_requires_an_offset_for_every_robot(robots):
  with pytest.raises(ValueError, match="robot_offsets_xy"):
    _make(robots, anchor_mode="static", robot_offsets_xy={1: [0.0, 0.0]})


@pytest.mark.slow
def test_gps_anchoring_measures_boot_enu_and_gates_frames(robots):
  """Frames are dropped until anchored, then boot_enu = enu(fix) - odom."""
  ds = _make(robots, anchor_mode="gps", anchor_samples=3)
  try:
    assert not ds.is_anchored(1) and not ds.is_anchored(2)

    odom1 = (7.0, 3.0, 12.0)
    odom2 = (-2.0, 1.0, 11.0)
    spawn1 = (0.0, 0.0)
    spawn2 = (100.0, 50.0)

    import threading
    stop = threading.Event()

    def pub():
      i = 0
      while not stop.is_set():
        t = 300.0 + i * 0.1
        robots[0].publish_frame(t, odom1)
        robots[1].publish_frame(t, odom2)
        robots[0].publish_fix(t, (spawn1[0] + odom1[0], spawn1[1] + odom1[1]))
        robots[1].publish_fix(t, (spawn2[0] + odom2[0], spawn2[1] + odom2[1]))
        i += 1
        time.sleep(0.03)

    th = threading.Thread(target=pub, daemon=True)
    th.start()
    assert rh.wait_until(lambda: ds.is_anchored(1) and ds.is_anchored(2),
                         timeout=40), "never anchored"
    np.testing.assert_allclose(ds.boot_enu(1)[:2], spawn1, atol=1e-2)
    np.testing.assert_allclose(ds.boot_enu(2)[:2], spawn2, atol=1e-2)
    # z is never anchored: it stays AGL.
    assert ds.boot_enu(1)[2] == 0.0 and ds.boot_enu(2)[2] == 0.0

    it = iter(ds)
    got = dict()
    deadline = time.time() + 40
    while len(got) < 2 and time.time() < deadline:
      f = next(it)
      got.setdefault(f["robot_id"], f)
    stop.set()
    th.join(timeout=5)
    assert set(got) == {1, 2}
    # Both robots must now report the SAME world position (they are 100/50 m
    # apart in world terms and their odom differs by exactly that).
    p2 = got[2]["pose_4x4"][:3, 3].numpy()
    expect_flu = np.array([spawn2[0] + odom2[0], spawn2[1] + odom2[1],
                           odom2[2]])
    np.testing.assert_allclose(
      p2, [-expect_flu[1], -expect_flu[2], expect_flu[0]], atol=1e-1)
  finally:
    ds.shutdown()


@pytest.mark.slow
def test_frame_skip_is_applied_per_robot(robots):
  """frame_skip=2 keeps 1 frame in 3, counted PER ROBOT."""
  # A queue big enough that nothing is dropped, so the count is exact.
  ds = _make(robots, anchor_mode="none", frame_skip=2, queue_size=500)
  try:
    _drive(robots, [(0, 0, 0), (0, 0, 0)], n=30, t0=400.0)
    time.sleep(2.0)
    for r in (1, 2):
      received = ds.streams[r].f
      queued = ds.streams[r].msgs.qsize()
      assert received >= 10, f"robot_{r} only saw {received} frames"
      assert queued == (received + 2) // 3, (received, queued)
  finally:
    ds.shutdown()


@pytest.mark.slow
def test_pose_stamped_mode(robots):
  """The legacy odom_to_pose_stamped bridge output is still accepted."""
  a = rh.FakeRobot(3, 92, use_pose_stamped=True)
  try:
    import threading
    stop = threading.Event()

    def pump():
      i = 0
      while not stop.is_set():
        a.info_pub.publish(a.camera_info())
        a.publish_frame(500.0 + i * 0.1, (1.0, 0.0, 2.0))
        i += 1
        time.sleep(0.05)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    ds = MultiRobotRos2Subscriber(
      robot_ids=[3], domain_ids=[92], pose_msg_type="pose_stamped",
      pose_topic="/{robot}/odometry_conversion/pose_stamped",
      anchor_mode="none", rgb_resolution=[16, 16], depth_resolution=[16, 16],
      intrinsics_timeout_s=30.0, sync_slop_s=0.05)
    try:
      f = next(iter(ds))
      np.testing.assert_allclose(f["pose_4x4"][:3, 3].numpy(),
                                 [0.0, -2.0, 1.0], atol=1e-4)
      assert f["robot_name"] == "robot_3"
    finally:
      stop.set()
      th.join(timeout=5)
      ds.shutdown()
  finally:
    a.shutdown()


def test_bad_configuration_is_rejected_early(robots):
  with pytest.raises(ValueError, match="anchor_mode"):
    _make(robots, anchor_mode="magic")
  with pytest.raises(ValueError, match="pose_msg_type"):
    _make(robots, pose_msg_type="tf")
  with pytest.raises(ValueError, match="Duplicate robot ids"):
    _make(robots, robot_ids=[1, 1], domain_ids=[D1, D2])
  with pytest.raises(ValueError, match="same length"):
    _make(robots, robot_ids=[1, 2], domain_ids=[D1])


@pytest.mark.slow
def test_shutdown_releases_only_its_own_contexts(robots):
  """One robot's teardown must not take another robot's context with it."""
  ds = _make(robots, anchor_mode="none")
  assert ros_context.refcount(D1) == 1
  assert ros_context.refcount(D2) == 1
  other = ros_context.acquire_context(D1)     # pretend the vis holds it too
  assert ros_context.refcount(D1) == 2

  ds.shutdown()
  # The dataset dropped its reference but the second holder keeps it alive.
  assert ros_context.refcount(D1) == 1
  assert other.ok()
  # D2 had only the dataset, so it is gone.
  assert ros_context.refcount(D2) == 0

  ros_context.release_context(other)
  assert ros_context.refcount(D1) == 0
  # The fake robots' own contexts are untouched by all of this.
  assert robots[0].context.ok() and robots[1].context.ok()


@pytest.mark.slow
def test_intrinsics_timeout_is_reported(robots):
  """No CameraInfo -> a clear timeout naming the topics, not a silent hang."""
  with pytest.raises(TimeoutError, match="CameraInfo"):
    MultiRobotRos2Subscriber(robot_ids=[7], domain_ids=[93],
                             intrinsics_timeout_s=3.0)
