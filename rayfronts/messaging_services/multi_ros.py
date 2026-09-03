"""Messaging service that fans one shared map out to several robot domains.

Every robot keeps talking to "its" rayfronts exactly as it does today: it
publishes text queries on ``/robot_i/rayfronts/msg_serv/new_text_query`` and
subscribes to ``/robot_i/rayfronts/msg_serv/{voxels_sim,rays_sim,frontiers}``.
What changed underneath is that there is now ONE map behind all of those, and
this class is the seam:

* **in** -- each robot's ``new_text_query`` (a label to add) and
  ``guiding_queries`` (that robot's CURRENT LVLM guiding list, JSON) land on
  that robot's own ROS domain and are merged into one shared query set. A
  guiding label lives while any robot lists it; a label that came in through
  ``new_text_query`` is never deleted (the original RAVEN ``delete_queries``
  rule).
* **out** -- query results, frontier clouds and the new status topic are
  published to EVERY robot's prefix, each copy translated back into THAT
  robot's local ``map`` frame, and only when something over there is actually
  subscribed.

Per-robot output topics and field layouts are byte-identical to the
single-robot ``Ros2MessagingService`` -- raven_nav cannot tell the difference.
"""

import logging
import threading
from functools import partial
from typing import Dict, Sequence

from typing_extensions import override

import numpy as np
import torch

from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
import std_msgs.msg

from rayfronts.messaging_services.base import MessagingService
from rayfronts.messaging_services.ros import (
  Ros2MessagingService, KEY_VOXEL_SIMILARITY, KEY_RAY_SIMILARITY,
  KEY_ALL_QUERIES)
from rayfronts import multi_robot_common as mrc

logger = logging.getLogger(__name__)

# RELIABLE + TRANSIENT_LOCAL depth 1: a late subscriber (semantic_search_task
# starting after the server, raven starting after that) must still see the
# current value rather than wait for the next tick.
LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=1)


class _ZeroAnchors:
  """Anchor source used before the dataset is wired in: no shift at all."""

  def boot_enu(self, robot_id):
    return np.zeros(3)

  def is_anchored(self, robot_id):
    return False


class MultiRobotRos2MessagingService(MessagingService):
  """One ``Ros2MessagingService`` per robot domain, driven by one mapper.

  Attributes:
    robot_ids: Robot ids served.
    domain_ids: Matching ROS domain ids.
    children: ``{robot_id: Ros2MessagingService}``.
    text_query_callback: Called with a single label string when any robot
      publishes on its ``new_text_query`` topic.
    guiding_callback: Called as ``(robot_id, [labels])`` when a robot publishes
      its guiding list.
  """

  def __init__(self,
               robot_ids: Sequence = (1,),
               domain_ids: Sequence = None,
               topic_prefix_template: str = "/{robot}/rayfronts/msg_serv",
               text_query_topic_template: str =
                 "/{robot}/rayfronts/msg_serv/new_text_query",
               guiding_queries_topic_template: str =
                 "/{robot}/rayfronts/msg_serv/guiding_queries",
               status_topic_template: str = "/{robot}/rayfronts/status",
               query_publish_threshold: float = 0.0,
               frame_id: str = "map",
               text_query_callback=None,
               guiding_callback=None):
    """
    Args:
      robot_ids: Robot ids to serve. Normally ``${dataset.robot_ids}``.
      domain_ids: ROS domain per robot. null = same as the robot id.
      topic_prefix_template: Prefix for this robot's published map topics.
      text_query_topic_template: Where this robot adds a query label.
      guiding_queries_topic_template: Where this robot publishes its full,
        current LVLM guiding list as a JSON array of strings.
      status_topic_template: Where the shared server reports per-robot state.
      query_publish_threshold: See Ros2MessagingService.
      frame_id: Frame id stamped on published clouds. Stays "map" -- the
        clouds are shifted into each robot's own map frame before publishing.
      text_query_callback: Called with the label string.
      guiding_callback: Called with ``(robot_id, [labels])``.
    """
    super().__init__()
    self.robot_ids = [int(r) for r in robot_ids]
    if domain_ids is None:
      self.domain_ids = list(self.robot_ids)
    else:
      self.domain_ids = [int(d) for d in domain_ids]
      if len(self.domain_ids) != len(self.robot_ids):
        raise ValueError("domain_ids must have the same length as robot_ids.")

    self.topic_prefix_template = topic_prefix_template
    self.text_query_topic_template = text_query_topic_template
    self.guiding_queries_topic_template = guiding_queries_topic_template
    self.status_topic_template = status_topic_template
    self.query_publish_threshold = query_publish_threshold
    self.frame_id = frame_id
    self.text_query_callback = text_query_callback
    self.guiding_callback = guiding_callback

    self._anchors = _ZeroAnchors()
    self._shift_cache: Dict[int, torch.Tensor] = dict()
    self._lock = threading.RLock()

    self.children: "Dict[int, Ros2MessagingService]" = dict()
    self.status_pubs = dict()
    self.guiding_subs = dict()
    for rid, did in zip(self.robot_ids, self.domain_ids):
      name = mrc.robot_name(rid)
      child = Ros2MessagingService(
        text_query_topic=mrc.fill_topic_template(text_query_topic_template,
                                                 rid),
        text_query_callback=partial(self._on_text_query, rid),
        query_publish_threshold=query_publish_threshold,
        topic_prefix=mrc.fill_topic_template(topic_prefix_template, rid),
        frame_id=frame_id,
        domain_id=did,
        node_name=f"rayfronts_msg_serv_{name}")
      self.children[rid] = child

      self.guiding_subs[rid] = child._rosnode.create_subscription(
        std_msgs.msg.String,
        mrc.fill_topic_template(guiding_queries_topic_template, rid),
        partial(self._on_guiding_queries, rid), LATCHED_QOS)
      self.status_pubs[rid] = child._rosnode.create_publisher(
        std_msgs.msg.String,
        mrc.fill_topic_template(status_topic_template, rid), LATCHED_QOS)

    logger.info("MultiRobotRos2MessagingService serving %s on domains %s.",
                [mrc.robot_name(r) for r in self.robot_ids], self.domain_ids)

  # ------------------------------------------------------------------ #
  # Anchors / frame shifting
  # ------------------------------------------------------------------ #

  def set_anchor_source(self, source):
    """Wire in the dataset (anything with ``boot_enu``/``is_anchored``)."""
    with self._lock:
      self._anchors = source if source is not None else _ZeroAnchors()
      self._shift_cache.clear()

  def world_to_local_shift(self, robot_id) -> np.ndarray:
    """RDF translation that maps shared-world coords into robot-local coords.

    ``boot_enu`` is a FLU/ENU offset; ``mrc.world_to_local_shift`` re-expresses
    ``-boot_enu`` in RDF, which for flu->rdf is ``(by, bz, -bx)``. Unit tested
    against ``geometry3d.get_coord_system_transform`` in
    ``tests/test_frame_shift.py``.
    """
    return mrc.world_to_local_shift(self._anchors.boot_enu(robot_id))

  def _shift_tensor(self, robot_id, like: torch.Tensor):
    """The RDF shift as a tensor, or None when it is exactly zero.

    Returning None for the zero case keeps the common single-robot-at-origin
    path free of an extra NxD add (and of a GPU sync to discover it is zero).
    """
    shift = self.world_to_local_shift(robot_id)
    if np.allclose(shift, 0.0):
      return None
    return torch.as_tensor(shift, dtype=like.dtype, device=like.device)

  # ------------------------------------------------------------------ #
  # Inbound
  # ------------------------------------------------------------------ #

  def _on_text_query(self, robot_id, data):
    """A robot added one label through its own new_text_query topic."""
    logger.info("[%s] new_text_query: %r", mrc.robot_name(robot_id), data)
    if self.text_query_callback is not None:
      self.text_query_callback(data)

  def _on_guiding_queries(self, robot_id, msg):
    labels = mrc.parse_guiding_payload(getattr(msg, "data", msg))
    logger.info("[%s] guiding_queries: %s", mrc.robot_name(robot_id), labels)
    if self.guiding_callback is not None:
      self.guiding_callback(robot_id, labels)

  @override
  def text_query_handler(self, s):
    """Only reached if someone calls this directly; children handle theirs."""
    data = getattr(s, "data", s)
    if self.text_query_callback is not None:
      self.text_query_callback(data)

  # ------------------------------------------------------------------ #
  # Outbound
  # ------------------------------------------------------------------ #

  def _needs(self, child, key, num_queries, query_labels) -> bool:
    """Does any of this robot's topics for ``key`` have a subscriber?"""
    base = f"{child.topic_prefix}/{key}"
    if child._has_subscriber(child._get_publisher(f"{base}/{KEY_ALL_QUERIES}")):
      return True
    for q in range(num_queries):
      suffix = mrc.query_topic_suffix(q, query_labels)
      if child._has_subscriber(child._get_publisher(f"{base}/{suffix}")):
        return True
    return False

  @override
  def publish_query_results(self, query_results: dict,
                            query_labels: list = None) -> None:
    """Publish the shared query result to every robot, in its own frame."""
    if not query_results:
      return
    has_vox = ("vox_xyz" in query_results and "vox_sim" in query_results
               and query_results["vox_xyz"] is not None)
    has_ray = ("ray_orig_angles" in query_results and "ray_sim" in query_results
               and query_results["ray_orig_angles"] is not None)
    if not (has_vox or has_ray):
      return

    n_vox_q = int(query_results["vox_sim"].shape[0]) if has_vox else 0
    n_ray_q = int(query_results["ray_sim"].shape[0]) if has_ray else 0

    for rid, child in self.children.items():
      want_vox = has_vox and self._needs(child, KEY_VOXEL_SIMILARITY,
                                         n_vox_q, query_labels)
      want_ray = has_ray and self._needs(child, KEY_RAY_SIMILARITY,
                                         n_ray_q, query_labels)
      if not (want_vox or want_ray):
        continue

      local = dict()
      if want_vox:
        vox_xyz = query_results["vox_xyz"]
        shift = self._shift_tensor(rid, vox_xyz)
        local["vox_xyz"] = vox_xyz if shift is None else vox_xyz + shift
        local["vox_sim"] = query_results["vox_sim"]
      if want_ray:
        roa = query_results["ray_orig_angles"]
        shift = self._shift_tensor(rid, roa)
        if shift is None:
          local["ray_orig_angles"] = roa
        else:
          # Only the ORIGIN moves; theta/phi are directions, a translation
          # leaves them alone.
          shifted = roa.clone()
          shifted[:, :3] = shifted[:, :3] + shift
          local["ray_orig_angles"] = shifted
        local["ray_sim"] = query_results["ray_sim"]

      child.publish_query_results(local, query_labels=query_labels)

  @override
  def publish_pc(self, pc_xyz: torch.FloatTensor,
                 features: Dict[str, torch.FloatTensor] = None,
                 layer: str = "pc") -> None:
    """Publish a map cloud (e.g. frontiers) to every robot, in its own frame."""
    if pc_xyz is None or pc_xyz.shape[0] == 0:
      return
    for rid, child in self.children.items():
      pub = child._get_publisher(f"{child.topic_prefix}/{layer}")
      if not child._has_subscriber(pub):
        continue
      shift = self._shift_tensor(rid, pc_xyz)
      local = pc_xyz if shift is None else pc_xyz + shift
      child.publish_pc(local, features=features, layer=layer)

  def publish_status(self, robot_id, status: dict) -> None:
    """Publish one robot's status JSON on ``/robot_i/rayfronts/status``."""
    pub = self.status_pubs.get(int(robot_id))
    if pub is None:
      return
    msg = std_msgs.msg.String()
    msg.data = mrc.status_to_json(status)
    pub.publish(msg)

  def status_topic(self, robot_id) -> str:
    return mrc.fill_topic_template(self.status_topic_template, robot_id)

  def guiding_topic(self, robot_id) -> str:
    return mrc.fill_topic_template(self.guiding_queries_topic_template,
                                   robot_id)

  def topic_prefix(self, robot_id) -> str:
    return mrc.fill_topic_template(self.topic_prefix_template, robot_id)

  # ------------------------------------------------------------------ #

  @override
  def join(self, timeout=None):
    for child in self.children.values():
      child.join(timeout)

  @override
  def shutdown(self):
    for child in self.children.values():
      try:
        child.shutdown()
      except Exception:
        logger.exception("Failed to shut down a messaging child.")
    logger.info("MultiRobotRos2MessagingService shutdown.")
