"""RViz/Foxglove visualisation of ONE shared map, mirrored per robot.

``MultiRobotRos2Vis`` owns one :class:`~rayfronts.visualizers.Ros2Vis` per robot
domain and splits the layers in two:

* **input layers** (``log_pose``, ``log_img``, ``log_depth_img``) describe the
  frame that was just consumed, so they go to the robot that produced it. The
  mapping server calls :meth:`set_active_robot` before logging inputs.
* **map layers** (``log_pc``, ``log_arrows`` and everything the base class
  builds on them -- heat clouds, feature clouds, occupancy, frontiers, per-query
  layers) describe the shared map, so every robot gets a copy, translated into
  its own local ``map`` frame.

Because the composite helpers in ``Mapping3DVisualizer`` (heat normalisation,
the PCA that colours feature clouds) run once here and only the primitives fan
out, all robots see the SAME colours for the same map -- which is the point of
sharing a map in the first place.
"""

import logging
import threading
from typing import Dict, Sequence

from typing_extensions import override

import numpy as np
import torch

try:
  from sensor_msgs.msg import PointCloud2
  from visualization_msgs.msg import MarkerArray
except ModuleNotFoundError:
  logging.getLogger(__name__).warning("ROS2 modules not found !")

from rayfronts.visualizers.base import Mapping3DVisualizer
from rayfronts.visualizers.ros import Ros2Vis
from rayfronts import feat_compressors
from rayfronts import multi_robot_common as mrc

logger = logging.getLogger(__name__)


class _ZeroAnchors:
  def boot_enu(self, robot_id):
    return np.zeros(3)

  def is_anchored(self, robot_id):
    return False


class MultiRobotRos2Vis(Mapping3DVisualizer):
  """Per-robot ROS2 visualisation of one shared map.

  Attributes:
    intrinsics_3x3: See base.
    img_size: See base.
    base_point_size: See base.
    global_heat_scale: See base.
    feat_compressor: See base.
    device: See base.
    time_step: See base.
    children: ``{robot_id: Ros2Vis}``.
    active_robot: Robot whose input layers are currently being logged.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               robot_ids: Sequence = (1,),
               domain_ids: Sequence = None,
               topic_prefix_template: str = "/{robot}/rayfronts",
               img_size=None,
               base_point_size: float = None,
               global_heat_scale: bool = False,
               feat_compressor: feat_compressors.FeatCompressor = None,
               reliability: str = "reliable",
               **kwargs):
    """
    Args:
      intrinsics_3x3: See base.
      robot_ids: Robot ids to publish to. Normally ``${dataset.robot_ids}``.
      domain_ids: ROS domain per robot. null = same as the robot id.
      topic_prefix_template: ``{robot}`` expands to ``robot_<id>``.
      img_size: See base.
      base_point_size: See base.
      global_heat_scale: See base.
      feat_compressor: See base.
      reliability: "reliable" (RViz2 needs it) or "best_effort".
    """
    super().__init__(intrinsics_3x3, img_size, base_point_size,
                     global_heat_scale, feat_compressor)

    self.robot_ids = [int(r) for r in robot_ids]
    if domain_ids is None:
      self.domain_ids = list(self.robot_ids)
    else:
      self.domain_ids = [int(d) for d in domain_ids]
      if len(self.domain_ids) != len(self.robot_ids):
        raise ValueError("domain_ids must have the same length as robot_ids.")
    self.topic_prefix_template = topic_prefix_template

    self._anchors = _ZeroAnchors()
    self._lock = threading.RLock()
    self.active_robot = self.robot_ids[0]

    self.children: "Dict[int, Ros2Vis]" = dict()
    for rid, did in zip(self.robot_ids, self.domain_ids):
      name = mrc.robot_name(rid)
      self.children[rid] = Ros2Vis(
        intrinsics_3x3=intrinsics_3x3,
        img_size=img_size,
        base_point_size=base_point_size,
        global_heat_scale=global_heat_scale,
        feat_compressor=self.feat_compressor,
        topic_prefix=mrc.fill_topic_template(topic_prefix_template, rid),
        reliability=reliability,
        domain_id=did,
        node_name=f"rayfronts_vis_{name}")
    logger.info("MultiRobotRos2Vis publishing to %s on domains %s.",
                [mrc.fill_topic_template(topic_prefix_template, r)
                 for r in self.robot_ids], self.domain_ids)

  # ------------------------------------------------------------------ #

  def set_anchor_source(self, source):
    """Wire in the dataset (anything with ``boot_enu``/``is_anchored``)."""
    with self._lock:
      self._anchors = source if source is not None else _ZeroAnchors()

  def set_active_robot(self, robot_id):
    """Route the next input-layer logs to this robot."""
    rid = int(robot_id)
    if rid in self.children:
      self.active_robot = rid

  def world_to_local_shift(self, robot_id) -> np.ndarray:
    return mrc.world_to_local_shift(self._anchors.boot_enu(robot_id))

  def _shift(self, robot_id, like: torch.Tensor):
    s = self.world_to_local_shift(robot_id)
    if np.allclose(s, 0.0):
      return None
    return torch.as_tensor(s, dtype=like.dtype, device=like.device)

  def _active_child(self) -> Ros2Vis:
    return self.children[self.active_robot]

  # ------------------------------------------------------------------ #
  # Input layers: only the robot that produced the frame.
  # ------------------------------------------------------------------ #

  @override
  def log_pose(self, pose_4x4: torch.FloatTensor, layer: str = "pose") -> None:
    self._active_child().log_pose(pose_4x4, layer=layer)

  @override
  def log_img(self, img: torch.FloatTensor, layer: str = "img",
              pose_layer: str = "pose") -> None:
    self._active_child().log_img(img, layer=layer, pose_layer=pose_layer)

  # ------------------------------------------------------------------ #
  # Map layers: every robot, shifted into its own frame.
  # ------------------------------------------------------------------ #

  @override
  def log_pc(self, pc_xyz: torch.FloatTensor,
             pc_rgb: torch.FloatTensor = None,
             pc_radii: torch.FloatTensor = None,
             layer: str = "pc"):
    if pc_xyz is None or pc_xyz.shape[0] == 0:
      return
    for rid, child in self.children.items():
      pub = child._get_publisher(layer, PointCloud2)
      if not child._has_subscriber(pub):
        continue
      shift = self._shift(rid, pc_xyz)
      local = pc_xyz if shift is None else pc_xyz + shift
      # Ros2Vis.log_pc premultiplies alpha IN PLACE when pc_rgb has 4 channels
      # (log_occ_pc feeds it exactly that). Fanning the same tensor out to N
      # children would premultiply it N times and every robot after the first
      # would see a darker map. Hand each child its own copy.
      child.log_pc(local, None if pc_rgb is None else pc_rgb.clone(),
                   pc_radii, layer=layer)

  @override
  def log_arrows(self, arr_origins, arr_dirs, arr_rgb=None, layer="arrows"):
    if arr_origins is None or arr_origins.shape[0] == 0:
      return
    for rid, child in self.children.items():
      pub = child._get_publisher(layer, MarkerArray)
      if not child._has_subscriber(pub):
        continue
      shift = self._shift(rid, arr_origins)
      local = arr_origins if shift is None else arr_origins + shift
      # Directions are unaffected by a translation.
      child.log_arrows(local, arr_dirs, arr_rgb, layer=layer)

  @override
  def log_box(self, box_mins, box_maxs, layer=""):
    # Ros2Vis.log_box is a stub upstream; keep the same no-op.
    pass

  # ------------------------------------------------------------------ #

  @override
  def step(self):
    super().step()
    for child in self.children.values():
      child.step()

  def shutdown(self):
    for child in self.children.values():
      try:
        child.shutdown()
      except Exception:
        logger.exception("Failed to shut down a vis child.")
    logger.info("MultiRobotRos2Vis shutdown.")
