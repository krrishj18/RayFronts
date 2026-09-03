"""local <-> world <-> RDF shift identities.

The shared map lives in one world frame; each robot's odometry lives in its own
``map`` frame; the mapper works in RDF. Getting the sign or the axis wrong here
puts robot_2's map 340 m from robot_1's and nothing downstream notices -- so
the shift is DERIVED here from ``get_coord_system_transform`` rather than
written down, and the numpy copy used on the hot path is checked against the
torch original.
"""

import numpy as np
import pytest

from conftest import HAVE_TORCH, ensure_pythonpath

# flu -> rdf: right = -left(y), down = -up(z), forward = +forward(x)
T_FLU2RDF = np.array([[0., -1., 0.],
                      [0., 0., -1.],
                      [1., 0., 0.]])


# --------------------------------------------------------------------------- #
# Pure numpy: runs anywhere.
# --------------------------------------------------------------------------- #

def test_flu_to_rdf_matrix(mrc):
  np.testing.assert_allclose(mrc.coord_system_transform("flu", "rdf"),
                             T_FLU2RDF)


def test_rdf_to_flu_is_the_inverse(mrc):
  a = mrc.coord_system_transform("flu", "rdf")
  b = mrc.coord_system_transform("rdf", "flu")
  np.testing.assert_allclose(a @ b, np.eye(3), atol=1e-12)
  # These are permutation-with-sign matrices, so the inverse is the transpose.
  np.testing.assert_allclose(b, a.T)


@pytest.mark.parametrize("src,tgt", [
  ("flu", "rdf"), ("rdf", "flu"), ("flu", "flu"), ("rdf", "rdf"),
  ("rfu", "rdf"), ("rdf", "rfu"), ("luf", "rdf"),
])
def test_transform_is_orthonormal(mrc, src, tgt):
  t = mrc.coord_system_transform(src, tgt)
  np.testing.assert_allclose(t @ t.T, np.eye(3), atol=1e-12)
  assert abs(abs(np.linalg.det(t)) - 1.0) < 1e-12


def test_offset_maps_axis_for_axis(mrc):
  # An offset of 10 m EAST (FLU +x) is 10 m along RDF +z (forward).
  np.testing.assert_allclose(mrc.transform_offset([10, 0, 0]), [0, 0, 10])
  # 10 m NORTH (FLU +y) is -10 m along RDF +x (right).
  np.testing.assert_allclose(mrc.transform_offset([0, 10, 0]), [-10, 0, 0])
  # 10 m UP (FLU +z) is -10 m along RDF +y (down).
  np.testing.assert_allclose(mrc.transform_offset([0, 0, 10]), [0, -10, 0])


def test_local_to_world_and_back_are_inverses(mrc):
  boot = [123.5, -47.25, 0.0]
  fwd = mrc.local_to_world_shift(boot)
  bwd = mrc.world_to_local_shift(boot)
  np.testing.assert_allclose(fwd + bwd, np.zeros(3), atol=1e-12)
  # Explicit expected values: (bx, by, 0) -> (-by, 0, bx) in RDF.
  np.testing.assert_allclose(fwd, [47.25, 0.0, 123.5])
  np.testing.assert_allclose(bwd, [-47.25, 0.0, -123.5])


def test_z_is_never_shifted(mrc):
  # boot_enu z is an MSL-datum difference, never applied. When it is 0 the RDF
  # y (down) component of the shift must be exactly 0.
  for boot in ([0, 0, 0], [10, 20, 0], [-500, 900, 0]):
    assert mrc.local_to_world_shift(boot)[1] == 0.0
    assert mrc.world_to_local_shift(boot)[1] == 0.0


def test_round_trip_on_a_point_cloud(mrc):
  rng = np.random.default_rng(0)
  pts_rdf_local = rng.normal(size=(64, 3)) * 20
  boot = [31.0, -12.0, 0.0]
  world = pts_rdf_local + mrc.local_to_world_shift(boot)
  back = world + mrc.world_to_local_shift(boot)
  np.testing.assert_allclose(back, pts_rdf_local, atol=1e-9)


def test_two_robots_agree_on_one_world_point(mrc):
  """The whole point: the same physical place is one world coordinate."""
  boot1 = [0.0, 0.0, 0.0]
  boot2 = [100.0, 50.0, 0.0]
  # A casualty 10 m east / 5 m north of robot_1's spawn, seen by both.
  local1_flu = np.array([10.0, 5.0, 0.0])
  local2_flu = local1_flu - np.array([100.0, 50.0, 0.0])  # robot_2's frame

  rdf1 = mrc.transform_offset(local1_flu) + mrc.local_to_world_shift(boot1)
  rdf2 = mrc.transform_offset(local2_flu) + mrc.local_to_world_shift(boot2)
  np.testing.assert_allclose(rdf1, rdf2, atol=1e-9)


def test_gps_to_enu_matches_frame_utils(mrc):
  # The stack-wide "Lisbon" origin must map to (0, 0, 0).
  x, y, z = mrc.gps_to_enu(mrc.DEFAULT_ORIGIN_LAT, mrc.DEFAULT_ORIGIN_LON,
                           mrc.DEFAULT_ORIGIN_ALT)
  assert (abs(x), abs(y), abs(z)) == (0.0, 0.0, 0.0)
  # 1 degree of latitude north is 111320 m by this flat-earth model.
  _, y1, _ = mrc.gps_to_enu(mrc.DEFAULT_ORIGIN_LAT + 1.0,
                            mrc.DEFAULT_ORIGIN_LON, mrc.DEFAULT_ORIGIN_ALT)
  assert abs(y1 - 111320.0) < 1e-6
  # Altitude is a plain offset from the origin altitude.
  _, _, z1 = mrc.gps_to_enu(mrc.DEFAULT_ORIGIN_LAT, mrc.DEFAULT_ORIGIN_LON,
                            mrc.DEFAULT_ORIGIN_ALT + 12.5)
  assert abs(z1 - 12.5) < 1e-9


def test_gps_anchor_recovers_a_known_spawn(mrc):
  """boot_enu = gps_to_enu(fix) - odom, exactly what the dataset computes."""
  spawn = np.array([120.0, -80.0])          # robot spawn in world ENU
  # Where the drone is right now, in its own map frame:
  odom = np.array([7.0, 3.0, 12.0])
  world = np.array([spawn[0] + odom[0], spawn[1] + odom[1], 0.0])
  # Invert gps_to_enu to build the fix the robot would report.
  import math
  lon = (world[0] / (111320.0 * math.cos(math.radians(mrc.DEFAULT_ORIGIN_LAT)))
         + mrc.DEFAULT_ORIGIN_LON)
  lat = world[1] / 111320.0 + mrc.DEFAULT_ORIGIN_LAT
  enu = np.array(mrc.gps_to_enu(lat, lon, mrc.DEFAULT_ORIGIN_ALT))
  boot = enu - odom
  np.testing.assert_allclose(boot[:2], spawn, atol=1e-6)


# --------------------------------------------------------------------------- #
# Against the real torch implementation.
# --------------------------------------------------------------------------- #

pytestmark_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")


@pytest.mark.torch
@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
@pytest.mark.parametrize("src,tgt", [
  ("flu", "rdf"), ("rdf", "flu"), ("rfu", "rdf"), ("luf", "rdf"),
  ("flu", "flu"), ("bdl", "rdf"),
])
def test_numpy_port_matches_geometry3d(mrc, src, tgt):
  ensure_pythonpath()
  from rayfronts import geometry3d as g3d
  ref = g3d.get_coord_system_transform(src, tgt).numpy()
  np.testing.assert_allclose(mrc.coord_system_transform(src, tgt), ref)


@pytest.mark.torch
@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_shift_commutes_with_the_rotation(mrc):
  """Shifting in FLU then rotating == rotating then shifting in RDF.

  This is the identity the dataset relies on (it shifts the pose translation in
  FLU) and the identity the messaging service relies on (it shifts published
  clouds in RDF). They must agree or robot_2's published map will be wrong by
  twice its offset.
  """
  ensure_pythonpath()
  import torch
  from rayfronts import geometry3d as g3d

  t = g3d.mat_3x3_to_4x4(g3d.get_coord_system_transform("flu", "rdf"))
  boot = torch.tensor([31.0, -12.0, 0.0])
  pts_flu = torch.randn(32, 3) * 25

  shifted_then_rotated = g3d.transform_points(pts_flu + boot, t)
  rotated_then_shifted = g3d.transform_points(pts_flu, t) + torch.as_tensor(
    mrc.local_to_world_shift(boot.numpy()), dtype=torch.float)
  torch.testing.assert_close(shifted_then_rotated, rotated_then_shifted,
                             atol=1e-4, rtol=1e-5)


@pytest.mark.torch
@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_pose_translation_lands_where_expected(mrc):
  """A pose shifted in FLU comes out of transform_pose_4x4 shifted in RDF."""
  ensure_pythonpath()
  import torch
  from rayfronts import geometry3d as g3d
  from scipy.spatial.transform import Rotation

  t = g3d.mat_3x3_to_4x4(g3d.get_coord_system_transform("flu", "rdf"))
  pose = torch.eye(4)
  pose[:3, :3] = torch.tensor(
    Rotation.from_euler("z", 37, degrees=True).as_matrix(), dtype=torch.float)
  pose[:3, 3] = torch.tensor([4.0, -9.0, 12.0])

  boot = np.array([31.0, -12.0, 0.0])
  shifted = pose.clone()
  shifted[0, 3] += boot[0]
  shifted[1, 3] += boot[1]

  rdf_plain = g3d.transform_pose_4x4(pose, t)
  rdf_shifted = g3d.transform_pose_4x4(shifted, t)

  delta = (rdf_shifted[:3, 3] - rdf_plain[:3, 3]).numpy()
  np.testing.assert_allclose(delta, mrc.local_to_world_shift(boot),
                             atol=1e-4)
  # Rotation must be untouched by a translation.
  torch.testing.assert_close(rdf_shifted[:3, :3], rdf_plain[:3, :3])
