"""MultiRobotRos2MessagingService: per-robot topics, frames, queries, status."""

import json
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
  import std_msgs.msg
  from sensor_msgs.msg import PointCloud2
  import ros_helpers as rh
  from rayfronts import ros_utils
  from rayfronts import multi_robot_common as mrc
  from rayfronts.messaging_services.multi_ros import (
    MultiRobotRos2MessagingService)

D1, D2 = 94, 95


class FakeAnchors:
  def __init__(self, table):
    self.table = {int(k): np.asarray(v, dtype=float) for k, v in table.items()}

  def boot_enu(self, robot_id):
    return self.table.get(int(robot_id), np.zeros(3))

  def is_anchored(self, robot_id):
    return int(robot_id) in self.table


@pytest.fixture
def service():
  seen_text = []
  seen_guiding = []
  svc = MultiRobotRos2MessagingService(
    robot_ids=[1, 2], domain_ids=[D1, D2],
    text_query_callback=seen_text.append,
    guiding_callback=lambda rid, labels: seen_guiding.append((rid, labels)))
  svc.seen_text = seen_text
  svc.seen_guiding = seen_guiding
  svc.set_anchor_source(FakeAnchors({1: [0.0, 0.0, 0.0],
                                     2: [100.0, 50.0, 0.0]}))
  try:
    yield svc
  finally:
    svc.shutdown()


def _query_results(n_vox=5, n_ray=3, n_q=2, device="cpu"):
  vox_xyz = torch.arange(n_vox * 3, dtype=torch.float,
                         device=device).reshape(n_vox, 3)
  vox_sim = torch.linspace(0.1, 0.9, n_q * n_vox,
                           device=device).reshape(n_q, n_vox)
  roa = torch.arange(n_ray * 5, dtype=torch.float,
                     device=device).reshape(n_ray, 5)
  ray_sim = torch.linspace(0.2, 0.8, n_q * n_ray,
                           device=device).reshape(n_q, n_ray)
  return dict(vox_xyz=vox_xyz, vox_sim=vox_sim,
              ray_orig_angles=roa, ray_sim=ray_sim)


# --------------------------------------------------------------------------- #
# Topic naming
# --------------------------------------------------------------------------- #

def test_per_robot_topic_names(service):
  assert service.topic_prefix(1) == "/robot_1/rayfronts/msg_serv"
  assert service.topic_prefix(2) == "/robot_2/rayfronts/msg_serv"
  assert service.status_topic(2) == "/robot_2/rayfronts/status"
  assert service.guiding_topic(1) == \
      "/robot_1/rayfronts/msg_serv/guiding_queries"
  assert service.children[1].text_query_topic == \
      "/robot_1/rayfronts/msg_serv/new_text_query"
  # Each child is on its own domain -> its own node.
  assert service.children[1]._rosnode.get_name() == "rayfronts_msg_serv_robot_1"
  assert service.children[2]._rosnode.get_name() == "rayfronts_msg_serv_robot_2"


def test_sanitized_per_query_topic_names(service):
  labels = ["person", "fallen tree"]
  service.publish_query_results(_query_results(n_q=2), query_labels=labels)
  names = set()
  for child in service.children.values():
    names.update(child._publishers.keys())
  assert "/robot_1/rayfronts/msg_serv/voxels_sim/q0_person" in names
  assert "/robot_1/rayfronts/msg_serv/voxels_sim/q1_fallen_tree" in names
  assert "/robot_2/rayfronts/msg_serv/rays_sim/q1_fallen_tree" in names
  assert "/robot_1/rayfronts/msg_serv/voxels_sim/all" in names


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_voxels_are_published_per_robot_in_that_robots_frame(service):
  s1 = rh.Sniffer("sniff1", D1, PointCloud2,
                  "/robot_1/rayfronts/msg_serv/voxels_sim/all")
  s2 = rh.Sniffer("sniff2", D2, PointCloud2,
                  "/robot_2/rayfronts/msg_serv/voxels_sim/all")
  try:
    qr = _query_results()
    assert rh.wait_until(
      lambda: (service.children[1]._get_publisher(
        "/robot_1/rayfronts/msg_serv/voxels_sim/all").get_subscription_count()
        and service.children[2]._get_publisher(
          "/robot_2/rayfronts/msg_serv/voxels_sim/all"
        ).get_subscription_count()), timeout=20), "sniffers never matched"

    def pump():
      service.publish_query_results(qr, query_labels=["person", "road"])

    m1 = s1.wait_for(1, timeout=20, pump=pump)
    m2 = s2.wait_for(1, timeout=20, pump=pump)
    assert m1 and m2

    a1 = ros_utils.pointcloud2_to_array(m1[-1])
    a2 = ros_utils.pointcloud2_to_array(m2[-1])
    # Frozen field layout: x, y, z, sim_0 ... sim_{Q-1}
    assert list(a1.dtype.names)[:3] == ["x", "y", "z"]
    assert "sim_0" in a1.dtype.names and "sim_1" in a1.dtype.names

    src = qr["vox_xyz"].numpy()
    got1 = np.stack([a1["x"], a1["y"], a1["z"]], axis=-1)
    got2 = np.stack([a2["x"], a2["y"], a2["z"]], axis=-1)
    # robot_1 is at the world origin -> no shift.
    np.testing.assert_allclose(got1, src, atol=1e-4)
    # robot_2's boot_enu is (100, 50, 0) FLU -> world->local RDF (50, 0, -100).
    expected = src + mrc.world_to_local_shift([100.0, 50.0, 0.0])
    np.testing.assert_allclose(got2, expected, atol=1e-3)
    # The similarity payload is identical for both robots.
    np.testing.assert_allclose(a1["sim_0"], a2["sim_0"], atol=1e-6)
  finally:
    s1.shutdown()
    s2.shutdown()


@pytest.mark.slow
def test_rays_shift_the_origin_but_not_the_angles(service):
  s2 = rh.Sniffer("sniffr2", D2, PointCloud2,
                  "/robot_2/rayfronts/msg_serv/rays_sim/all")
  try:
    qr = _query_results()
    assert rh.wait_until(
      lambda: service.children[2]._get_publisher(
        "/robot_2/rayfronts/msg_serv/rays_sim/all").get_subscription_count(),
      timeout=20)

    def pump():
      service.publish_query_results(qr, query_labels=["person", "road"])

    msgs = s2.wait_for(1, timeout=20, pump=pump)
    assert msgs
    a = ros_utils.pointcloud2_to_array(msgs[-1])
    assert list(a.dtype.names)[:5] == ["x", "y", "z", "theta", "phi"]
    src = qr["ray_orig_angles"].numpy()
    shift = mrc.world_to_local_shift([100.0, 50.0, 0.0])
    np.testing.assert_allclose(
      np.stack([a["x"], a["y"], a["z"]], -1), src[:, :3] + shift, atol=1e-3)
    # A translation does not rotate a direction.
    np.testing.assert_allclose(a["theta"], src[:, 3], atol=1e-4)
    np.testing.assert_allclose(a["phi"], src[:, 4], atol=1e-4)
  finally:
    s2.shutdown()


@pytest.mark.slow
def test_frontier_cloud_is_shifted_too(service):
  s2 = rh.Sniffer("snifff2", D2, PointCloud2,
                  "/robot_2/rayfronts/msg_serv/frontiers")
  try:
    pts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    feats = {"empty_cnt": torch.tensor([1.0, 2.0]),
             "unobserved_cnt": torch.tensor([3.0, 4.0]),
             "occupied_cnt": torch.tensor([5.0, 6.0])}
    assert rh.wait_until(
      lambda: service.children[2]._get_publisher(
        "/robot_2/rayfronts/msg_serv/frontiers").get_subscription_count(),
      timeout=20)

    msgs = s2.wait_for(
      1, timeout=20,
      pump=lambda: service.publish_pc(pts, features=feats, layer="frontiers"))
    assert msgs
    a = ros_utils.pointcloud2_to_array(msgs[-1])
    assert set(["x", "y", "z", "empty_cnt", "unobserved_cnt",
                "occupied_cnt"]).issubset(set(a.dtype.names))
    shift = mrc.world_to_local_shift([100.0, 50.0, 0.0])
    np.testing.assert_allclose(np.stack([a["x"], a["y"], a["z"]], -1),
                               pts.numpy() + shift, atol=1e-3)
    np.testing.assert_allclose(a["empty_cnt"], [1.0, 2.0])
  finally:
    s2.shutdown()


def test_nothing_is_published_without_a_subscriber(service):
  """The expensive GPU->CPU conversion must be skipped, not just the publish."""
  qr = _query_results()
  for child in service.children.values():
    for topic, pub in child._publishers.items():
      assert pub.get_subscription_count() == 0
  # Must not raise, and must leave no trace.
  service.publish_query_results(qr, query_labels=["person", "road"])
  service.publish_pc(torch.zeros(3, 3), layer="frontiers")


def test_empty_and_malformed_results_are_ignored(service):
  service.publish_query_results({}, query_labels=[])
  service.publish_query_results(None)
  service.publish_query_results({"vox_xyz": None, "vox_sim": None})
  service.publish_pc(torch.zeros(0, 3), layer="frontiers")


# --------------------------------------------------------------------------- #
# Inbound queries
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_new_text_query_reaches_the_callback(service):
  talker = rh.Talker("tq", D1, "/robot_1/rayfronts/msg_serv/new_text_query")
  try:
    deadline = time.time() + 25
    while not service.seen_text and time.time() < deadline:
      talker.say("person")
      time.sleep(0.1)
    assert "person" in service.seen_text
  finally:
    talker.shutdown()


@pytest.mark.slow
def test_guiding_queries_are_parsed_and_attributed_to_the_robot(service):
  t1 = rh.Talker("gq1", D1, "/robot_1/rayfronts/msg_serv/guiding_queries",
                 qos=rh.LATCHED)
  t2 = rh.Talker("gq2", D2, "/robot_2/rayfronts/msg_serv/guiding_queries",
                 qos=rh.LATCHED)
  try:
    deadline = time.time() + 25
    while time.time() < deadline:
      ids = {r for r, _ in service.seen_guiding}
      if {1, 2}.issubset(ids):
        break
      t1.say(json.dumps(["car", "mailbox"]))
      t2.say(json.dumps(["fence"]))
      time.sleep(0.2)
    by_robot = dict(service.seen_guiding)
    assert by_robot.get(1) == ["car", "mailbox"], service.seen_guiding
    assert by_robot.get(2) == ["fence"], service.seen_guiding
  finally:
    t1.shutdown()
    t2.shutdown()


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_status_json_reaches_each_robot_with_the_frozen_schema(service):
  s1 = rh.Sniffer("st1", D1, std_msgs.msg.String, "/robot_1/rayfronts/status",
                  qos=rh.LATCHED)
  s2 = rh.Sniffer("st2", D2, std_msgs.msg.String, "/robot_2/rayfronts/status",
                  qos=rh.LATCHED)
  try:
    def pump():
      for rid in (1, 2):
        service.publish_status(rid, mrc.build_status(
          robot_id=rid, domain_id=(D1 if rid == 1 else D2), anchored=True,
          boot_enu=[0.0, 0.0, 0.0] if rid == 1 else [100.0, 50.0, 0.0],
          frames_robot=3 * rid, frames_total=9, queries=["person", "road"],
          vox_count=42, ray_count=7, ts=time.time()))

    m1 = s1.wait_for(1, timeout=25, pump=pump)
    m2 = s2.wait_for(1, timeout=25, pump=pump)
    assert m1 and m2
    p1 = json.loads(m1[-1].data)
    p2 = json.loads(m2[-1].data)
    assert list(p1.keys()) == list(mrc.STATUS_KEYS)
    assert p1["robot"] == "robot_1" and p1["domain"] == D1
    assert p2["robot"] == "robot_2" and p2["boot_enu"] == [100.0, 50.0, 0.0]
    assert p1["frames_robot"] == 3 and p2["frames_robot"] == 6
    assert p1["queries"] == ["person", "road"]
    assert p1["vox_count"] == 42 and p1["ray_count"] == 7
    assert p1["anchored"] is True
  finally:
    s1.shutdown()
    s2.shutdown()


# --------------------------------------------------------------------------- #
# Shift helper
# --------------------------------------------------------------------------- #

def test_shift_matches_the_shared_helper(service):
  np.testing.assert_allclose(service.world_to_local_shift(1), np.zeros(3))
  np.testing.assert_allclose(
    service.world_to_local_shift(2),
    mrc.world_to_local_shift([100.0, 50.0, 0.0]))
  # (bx, by, 0) -> (by, 0, -bx)
  np.testing.assert_allclose(service.world_to_local_shift(2),
                             [50.0, 0.0, -100.0])


def test_no_anchor_source_means_no_shift():
  svc = MultiRobotRos2MessagingService(robot_ids=[1], domain_ids=[96])
  try:
    np.testing.assert_allclose(svc.world_to_local_shift(1), np.zeros(3))
  finally:
    svc.shutdown()
