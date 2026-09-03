"""MultiRobotRos2Vis: input layers follow the robot, map layers go to all."""

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
  from sensor_msgs.msg import Image, PointCloud2
  from geometry_msgs.msg import PoseStamped
  import ros_helpers as rh
  from rayfronts import ros_utils
  from rayfronts import multi_robot_common as mrc
  from rayfronts.visualizers.multi_ros import MultiRobotRos2Vis

D1, D2 = 97, 98


class FakeAnchors:
  def __init__(self, table):
    self.table = {int(k): np.asarray(v, dtype=float) for k, v in table.items()}

  def boot_enu(self, robot_id):
    return self.table.get(int(robot_id), np.zeros(3))

  def is_anchored(self, robot_id):
    return int(robot_id) in self.table


@pytest.fixture
def vis():
  intr = torch.tensor([[8.0, 0, 8.0], [0, 8.0, 8.0], [0, 0, 1.0]])
  v = MultiRobotRos2Vis(intrinsics_3x3=intr, robot_ids=[1, 2],
                        domain_ids=[D1, D2], base_point_size=0.25)
  v.set_anchor_source(FakeAnchors({1: [0.0, 0.0, 0.0],
                                   2: [100.0, 50.0, 0.0]}))
  try:
    yield v
  finally:
    v.shutdown()


def test_per_robot_prefixes_and_node_names(vis):
  assert vis.children[1].topic_prefix == "/robot_1/rayfronts"
  assert vis.children[2].topic_prefix == "/robot_2/rayfronts"
  assert vis.children[1]._rosnode.get_name() == "rayfronts_vis_robot_1"
  assert vis.children[2]._rosnode.get_name() == "rayfronts_vis_robot_2"
  # One shared feature compressor so both robots colour the map identically.
  assert vis.children[1].feat_compressor is vis.feat_compressor
  assert vis.children[2].feat_compressor is vis.feat_compressor


@pytest.mark.slow
def test_pose_goes_only_to_the_active_robot(vis):
  s1 = rh.Sniffer("vp1", D1, PoseStamped, "/robot_1/rayfronts/pose/pose")
  s2 = rh.Sniffer("vp2", D2, PoseStamped, "/robot_2/rayfronts/pose/pose")
  try:
    assert rh.wait_until(
      lambda: (vis.children[1]._get_publisher("pose/pose", PoseStamped)
               .get_subscription_count()
               and vis.children[2]._get_publisher("pose/pose", PoseStamped)
               .get_subscription_count()), timeout=20)

    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

    vis.set_active_robot(1)
    got = s1.wait_for(1, timeout=20, pump=lambda: vis.log_pose(pose))
    assert got, "robot_1 never got its own pose"
    assert s2.count() == 0, "robot_2 got a pose it did not produce"

    s1.clear()
    vis.set_active_robot(2)
    got2 = s2.wait_for(1, timeout=20, pump=lambda: vis.log_pose(pose))
    assert got2, "robot_2 never got its own pose"
    assert s1.count() == 0, "robot_1 got robot_2's pose"
  finally:
    s1.shutdown()
    s2.shutdown()


@pytest.mark.slow
def test_input_image_goes_only_to_the_active_robot(vis):
  s1 = rh.Sniffer("vi1", D1, Image, "/robot_1/rayfronts/pose/img")
  s2 = rh.Sniffer("vi2", D2, Image, "/robot_2/rayfronts/pose/img")
  try:
    assert rh.wait_until(
      lambda: (vis.children[1]._get_publisher("pose/img", Image)
               .get_subscription_count()
               and vis.children[2]._get_publisher("pose/img", Image)
               .get_subscription_count()), timeout=20)
    img = torch.rand(8, 8, 3)
    vis.set_active_robot(2)
    got = s2.wait_for(1, timeout=20, pump=lambda: vis.log_img(img))
    assert got
    assert s1.count() == 0
  finally:
    s1.shutdown()
    s2.shutdown()


@pytest.mark.slow
def test_map_cloud_goes_to_every_robot_shifted(vis):
  s1 = rh.Sniffer("vm1", D1, PointCloud2, "/robot_1/rayfronts/voxel_rgb")
  s2 = rh.Sniffer("vm2", D2, PointCloud2, "/robot_2/rayfronts/voxel_rgb")
  try:
    assert rh.wait_until(
      lambda: (vis.children[1]._get_publisher("voxel_rgb", PointCloud2)
               .get_subscription_count()
               and vis.children[2]._get_publisher("voxel_rgb", PointCloud2)
               .get_subscription_count()), timeout=20)

    pc = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def pump():
      vis.log_pc(pc, rgb, layer="voxel_rgb")

    m1 = s1.wait_for(1, timeout=20, pump=pump)
    m2 = s2.wait_for(1, timeout=20, pump=pump)
    assert m1 and m2

    a1 = ros_utils.split_rgb_field(ros_utils.pointcloud2_to_array(m1[-1]))
    a2 = ros_utils.split_rgb_field(ros_utils.pointcloud2_to_array(m2[-1]))
    got1 = np.stack([a1["x"], a1["y"], a1["z"]], -1)
    got2 = np.stack([a2["x"], a2["y"], a2["z"]], -1)

    # Ros2Vis publishes in FLU, so undo its rdf->flu rotation to compare.
    rdf2flu = mrc.coord_system_transform("rdf", "flu")
    back1 = got1 @ rdf2flu           # (T @ p)^T == p^T @ T^T; T is orthonormal
    back2 = got2 @ rdf2flu
    np.testing.assert_allclose(back1, pc.numpy(), atol=1e-3)
    expected2 = pc.numpy() + mrc.world_to_local_shift([100.0, 50.0, 0.0])
    np.testing.assert_allclose(back2, expected2, atol=1e-3)
  finally:
    s1.shutdown()
    s2.shutdown()


@pytest.mark.slow
def test_heat_layer_from_the_base_class_fans_out(vis):
  """log_heat_pc is a base-class composite built on log_pc; it must fan out."""
  s1 = rh.Sniffer("vh1", D1, PointCloud2,
                  "/robot_1/rayfronts/queries/person/voxels")
  s2 = rh.Sniffer("vh2", D2, PointCloud2,
                  "/robot_2/rayfronts/queries/person/voxels")
  try:
    assert rh.wait_until(
      lambda: (vis.children[1]._get_publisher("queries/person/voxels",
                                              PointCloud2)
               .get_subscription_count()
               and vis.children[2]._get_publisher("queries/person/voxels",
                                                  PointCloud2)
               .get_subscription_count()), timeout=20)
    pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    heat = torch.tensor([0.1, 0.5, 0.9])
    m1 = s1.wait_for(1, timeout=20, pump=lambda: vis.log_heat_pc(
      pc, heat, layer="queries/person/voxels"))
    m2 = s2.wait_for(1, timeout=20, pump=lambda: vis.log_heat_pc(
      pc, heat, layer="queries/person/voxels"))
    assert m1 and m2
  finally:
    s1.shutdown()
    s2.shutdown()


@pytest.mark.slow
def test_alpha_premultiply_is_not_applied_twice(vis):
  """Ros2Vis.log_pc premultiplies a 4-channel colour IN PLACE.

  Fanning ONE tensor out to N children would premultiply it N times and every
  robot after the first would get a darker map, so each child gets a copy.
  log_occ_pc (which the mapper uses for the voxel_occ layer) is exactly the
  caller that supplies a 4-channel colour.
  """
  s1 = rh.Sniffer("va1", D1, PointCloud2, "/robot_1/rayfronts/voxel_occ")
  s2 = rh.Sniffer("va2", D2, PointCloud2, "/robot_2/rayfronts/voxel_occ")
  try:
    assert rh.wait_until(
      lambda: (vis.children[1]._get_publisher("voxel_occ", PointCloud2)
               .get_subscription_count()
               and vis.children[2]._get_publisher("voxel_occ", PointCloud2)
               .get_subscription_count()), timeout=20)

    pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    # (N, 1), the shape SemanticRayFrontiersMap.vis_map actually passes.
    occ = torch.tensor([[0.0], [1.0]])

    def pump():
      vis.log_occ_pc(pc, occ, layer="voxel_occ")

    m1 = s1.wait_for(1, timeout=20, pump=pump)
    m2 = s2.wait_for(1, timeout=20, pump=pump)
    assert m1 and m2
    a1 = ros_utils.split_rgb_field(ros_utils.pointcloud2_to_array(m1[-1]))
    a2 = ros_utils.split_rgb_field(ros_utils.pointcloud2_to_array(m2[-1]))
    for ch in ("r", "g", "b"):
      np.testing.assert_array_equal(a1[ch], a2[ch])
    # log_occ_pc's colour is (0.8, 0.2, 0.8) scaled by alpha; the fully
    # occupied point must keep full strength on BOTH robots.
    assert int(a1["r"].max()) == int(a2["r"].max()) > 100
  finally:
    s1.shutdown()
    s2.shutdown()


@pytest.mark.slow
def test_the_callers_colour_tensor_is_not_mutated(vis):
  """The single-robot Ros2Vis mutates its argument; the fan-out must not."""
  s1 = rh.Sniffer("vc1", D1, PointCloud2, "/robot_1/rayfronts/rgba_layer")
  try:
    assert rh.wait_until(
      lambda: vis.children[1]._get_publisher("rgba_layer", PointCloud2)
      .get_subscription_count(), timeout=20)
    pc = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    rgba = torch.tensor([[1.0, 1.0, 1.0, 0.5], [1.0, 0.0, 0.0, 1.0]])
    before = rgba.clone()
    msgs = s1.wait_for(1, timeout=20,
                       pump=lambda: vis.log_pc(pc, rgba, layer="rgba_layer"))
    assert msgs
    torch.testing.assert_close(rgba, before)
  finally:
    s1.shutdown()


def test_nothing_published_without_a_subscriber(vis):
  vis.log_pc(torch.rand(4, 3), layer="voxel_rgb")
  vis.log_arrows(torch.rand(2, 3), torch.rand(2, 3), layer="rays")
  vis.log_pc(torch.zeros(0, 3), layer="voxel_rgb")
  vis.log_box(None, None)
  vis.step()


def test_active_robot_is_clamped_to_known_robots(vis):
  vis.set_active_robot(2)
  assert vis.active_robot == 2
  vis.set_active_robot(99)          # unknown -> unchanged, no crash
  assert vis.active_robot == 2
