"""One posed-RGBD stream per robot, each on its own ROS domain, round-robined.

``MultiRobotRos2Subscriber`` is what lets ONE ``SemanticRayFrontiersMap`` cover
several robots.  Every robot publishes on its own ``ROS_DOMAIN_ID`` (isaac-sim
spawns robot *i* with ``domain_id=i``) and stamps its odometry in its OWN ``map``
frame, anchored at that robot's spawn point -- robot_1's (0,0) and robot_2's
(0,0) are different places.  This class therefore does three things the
single-robot ``Ros2Subscriber`` does not:

1. **Per-domain contexts.**  One private ``rclpy.Context`` per robot domain
   (see :mod:`rayfronts.ros_context`), one node each, one spinner thread each.
2. **Anchoring.**  Each robot's ``boot_enu`` -- the ENU position of its ``map``
   origin -- is measured as ``mean(gps_to_enu(fix) - odom_xyz)`` over
   ``anchor_samples`` fixes, exactly the recipe ``map_anchor_node`` and
   ``raven_nav._navsat_cb`` use, against the same fixed "Lisbon" world origin.
   ``static`` (offsets from the config, e.g. straight out of ``SPAWN_CONFIGS``)
   and ``none`` (single robot / already-shared frame) are also supported.
3. **Local -> world before FLU -> RDF.**  The xy shift is applied to the pose
   *translation while it is still in FLU*, then the whole pose is rotated into
   the RDF convention the mapper wants.  z is left alone: every robot's z is
   AGL, and boot_enu's z is an MSL-datum difference that must not be added
   (same rule as ``raven_nav._local_to_world``).

Frames from all robots are yielded round-robin, so no robot can starve the map
while another is flying fast.
"""

import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)

import numpy as np
import torch

try:
  import rclpy
  from rclpy.node import Node
  from rclpy.executors import SingleThreadedExecutor
  from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
  import message_filters
  from sensor_msgs.msg import Image, CameraInfo, NavSatFix
  from geometry_msgs.msg import PoseStamped
  from nav_msgs.msg import Odometry
  from rayfronts.ros_utils import image_to_numpy, pose_to_numpy
except ModuleNotFoundError:
  logger.warning("ROS2 modules not found !")

from rayfronts.datasets.base import PosedRgbdDataset
from rayfronts import geometry3d as g3d
from rayfronts import ros_context
from rayfronts import multi_robot_common as mrc

ANCHOR_MODES = ("gps", "static", "none")
POSE_MSG_TYPES = ("odometry", "pose_stamped")


class _RobotStream:
  """One robot: its own context, node, executor thread and frame queue."""

  def __init__(self, parent, robot_id, domain_id):
    self.parent = parent
    self.robot_id = robot_id
    self.robot_name = mrc.robot_name(robot_id)
    self.domain_id = int(domain_id)

    self.f = 0                       # frames received (pre frame_skip)
    self.frames_yielded = 0          # frames handed to the mapper
    self.frames_dropped_unanchored = 0
    self.msgs = queue.Queue(maxsize=parent.queue_size)

    self._pose_lock = threading.Lock()
    self._last_pose_xyz = None       # latest FLU translation, robot-local
    self._first_z = None             # first odometry z (start_after_climb_m)
    self.airborne = parent.start_after_climb_m <= 0.0
    self.frames_dropped_grounded = 0
    self._anchor_samples: List[np.ndarray] = []
    self.boot_enu = np.zeros(3, dtype=np.float64)
    self.anchored = parent.anchor_mode != "gps"
    self._logged_wait = False

    if parent.anchor_mode == "static":
      off = parent.robot_offsets_xy.get(str(robot_id))
      if off is None:
        raise ValueError(
          f"anchor_mode=static but robot_offsets_xy has no entry for "
          f"{robot_id!r}. Got keys {sorted(parent.robot_offsets_xy)}.")
      self.boot_enu = np.array([float(off[0]), float(off[1]), 0.0])

    self.context = ros_context.acquire_context(self.domain_id)
    self.node = None
    self.executor = None
    try:
      self._build(parent, robot_id)
    except Exception:
      # Never leak a context reference on a half-built stream: the next
      # component asking for this domain would inherit a phantom refcount and
      # the context would outlive the process's use of it.
      self.shutdown()
      raise

  def _build(self, parent, robot_id):
    self.node = Node(f"rayfronts_input_{self.robot_name}",
                     context=self.context)

    t = parent.topics_for(robot_id)
    self.topics = t

    self.intrinsics_sub = self.node.create_subscription(
      CameraInfo, t["intrinsics"], self._on_camera_info,
      QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1))

    pose_type = Odometry if parent.pose_msg_type == "odometry" else PoseStamped
    self._subs = OrderedDict()
    self._subs["rgb"] = message_filters.Subscriber(
      self.node, Image, t["rgb"], qos_profile=parent.sensor_qos())
    self._subs["depth"] = message_filters.Subscriber(
      self.node, Image, t["depth"], qos_profile=parent.sensor_qos())
    self._subs["pose"] = message_filters.Subscriber(
      self.node, pose_type, t["pose"], qos_profile=parent.sensor_qos())

    self._sync = message_filters.ApproximateTimeSynchronizer(
      list(self._subs.values()), queue_size=parent.queue_size,
      slop=parent.sync_slop_s, allow_headerless=False)
    self._sync.registerCallback(self._on_frame)

    # A second, plain subscription on the pose topic so anchoring does not have
    # to wait for a synchronised rgb+depth triple (the camera may still be
    # spinning up while GPS fixes are already flowing).
    self.node.create_subscription(
      pose_type, t["pose"], self._on_pose,
      QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1))

    if parent.anchor_mode == "gps":
      if not t.get("navsat"):
        raise ValueError("anchor_mode=gps requires a navsat_topic template.")
      self.node.create_subscription(
        NavSatFix, t["navsat"], self._on_navsat,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                   history=HistoryPolicy.KEEP_LAST, depth=1))

    self.executor = SingleThreadedExecutor(context=self.context)
    self.executor.add_node(self.node)
    self.spin_thread = threading.Thread(
      target=self._spin, name=f"rayfronts_input_{self.robot_name}_spinner",
      daemon=True)
    self.spin_thread.start()

  # -- ROS callbacks -------------------------------------------------------- #

  def _spin(self):
    try:
      self.executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass
    except Exception:  # pragma: no cover - defensive
      logger.exception("Spinner for %s died.", self.robot_name)

  def _on_camera_info(self, msg):
    self.parent._register_intrinsics(self, msg)

  @staticmethod
  def _pose_of(msg):
    return msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose

  def _on_pose(self, msg):
    p = self._pose_of(msg).position
    with self._pose_lock:
      self._last_pose_xyz = np.array([p.x, p.y, p.z], dtype=np.float64)
    if not self.airborne:
      # ABSOLUTE local z, not climb-since-first-sample: odometry z in this
      # stack is AGL-ish (spawn ~1 m). A mapper that (re)starts on an
      # already-airborne drone must begin mapping immediately — the
      # climb-delta form wedged incarnation #2 live (2026-09-02 23:20,
      # "dropping frames until the robot climbs 3.0 m" forever at 8.5 m).
      if float(p.z) >= self.parent.start_after_climb_m:
        self.airborne = True
        logger.info("[%s] airborne (z=%.1f >= %.1f m) — starting to map.",
                    self.robot_name, float(p.z),
                    self.parent.start_after_climb_m)

  def _on_navsat(self, msg):
    if self.anchored:
      return
    lat, lon, alt = float(msg.latitude), float(msg.longitude), \
        float(msg.altitude)
    if not np.isfinite([lat, lon, alt]).all():
      return
    with self._pose_lock:
      odom = None if self._last_pose_xyz is None else self._last_pose_xyz.copy()
    if odom is None:
      if not self._logged_wait:
        logger.info("[%s] GPS fix received but no odometry yet; waiting.",
                    self.robot_name)
        self._logged_wait = True
      return
    enu = np.array(mrc.gps_to_enu(lat, lon, alt,
                                  *self.parent.gps_origin), dtype=np.float64)
    # The ORIGIN of this robot's map frame in world ENU, not where the drone
    # happens to be right now: raven starts after takeoff.
    self._anchor_samples.append(enu - odom)
    n = len(self._anchor_samples)
    if n >= self.parent.anchor_samples:
      mean = np.mean(np.stack(self._anchor_samples, axis=0), axis=0)
      # xy only: z stays AGL (boot_enu[2] is an MSL-datum offset).
      self.boot_enu = np.array([mean[0], mean[1], 0.0])
      self.anchored = True
      logger.info("[%s] anchored after %d fixes: boot_enu=(%.2f, %.2f) "
                  "(measured z offset %.2f m, not applied).",
                  self.robot_name, n, self.boot_enu[0], self.boot_enu[1],
                  float(mean[2]))

  def _on_frame(self, *msgs):
    if self.parent.frame_skip <= 0 or \
        self.f % (self.parent.frame_skip + 1) == 0:
      if not self.anchored:
        self.frames_dropped_unanchored += 1
        if self.frames_dropped_unanchored == 1:
          logger.info("[%s] dropping frames until the robot is anchored "
                      "(anchor_mode=%s).", self.robot_name,
                      self.parent.anchor_mode)
      elif not self.airborne:
        self.frames_dropped_grounded += 1
        if self.frames_dropped_grounded == 1:
          logger.info("[%s] dropping frames until the robot climbs %.1f m "
                      "(start_after_climb_m).", self.robot_name,
                      self.parent.start_after_climb_m)
      else:
        if self.msgs.full():
          try:
            self.msgs.get_nowait()  # discard oldest, prioritise fresh data
          except queue.Empty:
            pass
        self.msgs.put(dict(zip(self._subs.keys(), msgs)))
        self.parent._wake()
    self.f += 1

  # -- teardown -------------------------------------------------------------- #

  def shutdown(self):
    if self.executor is not None and self.node is not None:
      try:
        self.executor.remove_node(self.node)
      except Exception:
        pass
    if self.node is not None:
      try:
        self.node.destroy_node()
      except Exception:
        logger.exception("[%s] failed to destroy node.", self.robot_name)
      self.node = None
    if self.context is not None:
      ros_context.release_context(self.context)
      self.context = None


class MultiRobotRos2Subscriber(PosedRgbdDataset):
  """Round-robin posed-RGBD source spanning several robots / ROS domains.

  Each yielded frame dict carries the usual ``rgb_img``/``depth_img``/
  ``pose_4x4`` plus ``robot_id`` (int) and ``robot_name`` (str) so the mapping
  server knows which robot produced it (for input visualisation and for the
  per-robot status topic).  Poses are already in the SHARED world frame and in
  RDF, so the mapper needs no knowledge of any of this.

  Attributes:
    intrinsics_3x3: See base. Taken from the first robot to publish CameraInfo;
      every other robot is checked against it.
    robot_ids: List of robot ids being consumed.
    domain_ids: Matching ROS domain ids.
    frames_total: Total frames handed to the mapper across all robots.
  """

  def __init__(self,
               robot_ids: Sequence = (1,),
               domain_ids: Sequence = None,
               rgb_topic: str = "/{robot}/sensors/front_stereo/left/image_rect",
               depth_topic: str =
                 "/{robot}/sensors/front_stereo/left/depth_ground_truth",
               intrinsics_topic: str =
                 "/{robot}/sensors/front_stereo/left/camera_info",
               pose_topic: str = "/{robot}/odometry_conversion/odometry",
               pose_msg_type: str = "odometry",
               navsat_topic: str =
                 "/{robot}/interface/mavros/global_position/global",
               anchor_mode: str = "gps",
               anchor_samples: int = 10,
               robot_offsets_xy=None,
               gps_origin: Sequence[float] = None,
               src_coord_system: str = "flu",
               rgb_resolution=None,
               depth_resolution=None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear",
               queue_size: int = 10,
               sync_slop_s: float = 0.01,
               sensor_reliability: str = "reliable",
               intrinsics_timeout_s: float = None,
               start_after_climb_m: float = 0.0):
    """
    Args:
      robot_ids: Robot ids to subscribe to. ``[1, 2]`` -> ``robot_1``,
        ``robot_2``.
      domain_ids: ROS_DOMAIN_ID per robot. ``null`` means "same as the robot
        id", which is what isaac-sim's multi-drone spawner does.
      rgb_topic/depth_topic/intrinsics_topic/pose_topic/navsat_topic: Topic
        TEMPLATES. ``{robot}`` expands to ``robot_<id>`` and ``{id}`` to the
        bare id.
      pose_msg_type: "odometry" (nav_msgs/Odometry, what
        ``odometry_conversion`` publishes) or "pose_stamped"
        (geometry_msgs/PoseStamped, what the legacy
        ``odom_to_pose_stamped`` bridge publishes).
      anchor_mode: How to find each robot's ``boot_enu``.
        * "gps"    -- measure it from NavSatFix + odometry (the real thing).
        * "static" -- take it from ``robot_offsets_xy`` (e.g. SPAWN_CONFIGS).
        * "none"   -- no shift at all; every robot is assumed to already share
          one frame. Correct for a single robot spawned at the world origin.
      anchor_samples: How many GPS fixes to average before freezing boot_enu.
      robot_offsets_xy: ``{id: [x, y]}`` for anchor_mode=static.
      gps_origin: ``[lat, lon, alt]`` of the shared world origin. null uses the
        stack-wide "Lisbon" constant, which is what every other component uses.
      src_coord_system: Convention the incoming poses are in. "flu" for
        AirStack odometry.
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base. Applied PER ROBOT.
      interp_mode: See base.
      queue_size: Per-robot frame queue depth (oldest is dropped when full).
      sync_slop_s: ApproximateTimeSynchronizer slop. 0.01 matches the proven
        single-robot deployment; raise it only if frames are being starved,
        since a looser slop pairs an image with a staler pose.
      sensor_reliability: QoS for rgb/depth/pose: "reliable" (matches the
        single-robot deployment) or "best_effort".
      intrinsics_timeout_s: Give up waiting for the first CameraInfo after this
        many seconds. null waits forever (the legacy behaviour).
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    if anchor_mode not in ANCHOR_MODES:
      raise ValueError(f"anchor_mode must be one of {ANCHOR_MODES}, "
                       f"got {anchor_mode!r}")
    if pose_msg_type not in POSE_MSG_TYPES:
      raise ValueError(f"pose_msg_type must be one of {POSE_MSG_TYPES}, "
                       f"got {pose_msg_type!r}")

    self.robot_ids = [int(r) for r in robot_ids]
    if len(self.robot_ids) == 0:
      raise ValueError("robot_ids must not be empty.")
    if len(set(self.robot_ids)) != len(self.robot_ids):
      raise ValueError(f"Duplicate robot ids: {self.robot_ids}")
    if domain_ids is None:
      self.domain_ids = list(self.robot_ids)
    else:
      self.domain_ids = [int(d) for d in domain_ids]
      if len(self.domain_ids) != len(self.robot_ids):
        raise ValueError("domain_ids must have the same length as robot_ids.")

    self._templates = dict(rgb=rgb_topic, depth=depth_topic,
                           intrinsics=intrinsics_topic, pose=pose_topic,
                           navsat=navsat_topic)
    self.pose_msg_type = pose_msg_type
    self.anchor_mode = anchor_mode
    self.anchor_samples = max(1, int(anchor_samples))
    # "rayfronts should start with raven, after takeoff" (user 2026-09-02):
    # frames are dropped until the robot has CLIMBED this much above its
    # first odometry sample. 0 = off (map from the first frame, the old
    # behaviour). Ground-level pre-takeoff frames were integrating junk
    # geometry (the underground artifact class) and pre-search clutter.
    self.start_after_climb_m = float(start_after_climb_m or 0.0)
    self.robot_offsets_xy = _normalize_offsets(robot_offsets_xy)
    self.gps_origin = tuple(gps_origin) if gps_origin else (
      mrc.DEFAULT_ORIGIN_LAT, mrc.DEFAULT_ORIGIN_LON, mrc.DEFAULT_ORIGIN_ALT)
    self.src_coord_system = src_coord_system
    self.queue_size = max(1, int(queue_size))
    self.sync_slop_s = float(sync_slop_s)
    self.sensor_reliability = sensor_reliability
    self.frames_total = 0

    self.src2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform(src_coord_system, "rdf"))

    self._shutdown_event = threading.Event()
    self._data_event = threading.Event()
    self._intrinsics_cond = threading.Condition()
    self._intrinsics_owner = None
    self._intrinsics_warned = set()
    self._rr_index = 0

    self.streams: "OrderedDict[int, _RobotStream]" = OrderedDict()
    try:
      for rid, did in zip(self.robot_ids, self.domain_ids):
        self.streams[rid] = _RobotStream(self, rid, did)
    except Exception:
      # A bad config for robot N must not leave robots 1..N-1 holding open
      # contexts and spinner threads for the rest of the process's life.
      self.shutdown()
      raise

    self._wait_for_intrinsics(intrinsics_timeout_s)
    logger.info("MultiRobotRos2Subscriber ready for %s on domains %s "
                "(anchor_mode=%s).",
                [mrc.robot_name(r) for r in self.robot_ids],
                self.domain_ids, self.anchor_mode)

  # -- configuration helpers ------------------------------------------------ #

  def topics_for(self, robot_id) -> Dict[str, str]:
    return {k: mrc.fill_topic_template(v, robot_id)
            for k, v in self._templates.items()}

  def sensor_qos(self):
    if self.sensor_reliability == "best_effort":
      return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=self.queue_size)
    # Matches the legacy Ros2Subscriber, which passes a bare depth int
    # (RELIABLE, KEEP_LAST) to message_filters.
    return self.queue_size

  def _wake(self):
    self._data_event.set()

  # -- intrinsics ------------------------------------------------------------ #

  def _register_intrinsics(self, stream, msg):
    k = torch.tensor(msg.k, dtype=torch.float).reshape(3, 3)
    with self._intrinsics_cond:
      if self.intrinsics_3x3 is None:
        self.original_h = msg.height
        self.original_w = msg.width
        self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
        self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
        self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
        self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w
        if (self.depth_h != self.original_h or
            self.depth_w != self.original_w):
          h_ratio = self.depth_h / self.original_h
          w_ratio = self.depth_w / self.original_w
          k[0, :] = k[0, :] * w_ratio
          k[1, :] = k[1, :] * h_ratio
        self.intrinsics_3x3 = k
        self._intrinsics_owner = stream.robot_id
        logger.info("Loaded intrinsics from %s (%dx%d):\n%s",
                    stream.robot_name, msg.width, msg.height, str(k))
        self._intrinsics_cond.notify_all()
      else:
        # Every robot must agree: one mapper, one intrinsics matrix.
        same_res = (msg.height == self.original_h and
                    msg.width == self.original_w)
        raw = torch.tensor(msg.k, dtype=torch.float).reshape(3, 3)
        ref = self.intrinsics_3x3
        if (self.depth_h != self.original_h or
            self.depth_w != self.original_w):
          raw[0, :] = raw[0, :] * (self.depth_w / max(1, msg.width))
          raw[1, :] = raw[1, :] * (self.depth_h / max(1, msg.height))
        same_k = bool(torch.allclose(raw, ref, atol=1e-3))
        if (not same_res or not same_k) and \
            stream.robot_id not in self._intrinsics_warned:
          self._intrinsics_warned.add(stream.robot_id)
          logger.error(
            "%s publishes different camera intrinsics/resolution than %s "
            "(%dx%d vs %dx%d).\n%s\nvs\n%s\nOne shared map needs ONE camera "
            "model; the mapper will keep using %s's. Fix ZED_WIDTH/"
            "ZED_HEIGHT so every robot matches.",
            stream.robot_name, mrc.robot_name(self._intrinsics_owner),
            msg.width, msg.height, self.original_w, self.original_h,
            str(raw), str(ref), mrc.robot_name(self._intrinsics_owner))
      try:
        stream.node.destroy_subscription(stream.intrinsics_sub)
      except Exception:
        pass

  def _wait_for_intrinsics(self, timeout_s):
    t0 = time.time()
    with self._intrinsics_cond:
      logged = False
      while self.intrinsics_3x3 is None:
        if self._shutdown_event.is_set():
          raise RuntimeError("Shut down while waiting for intrinsics.")
        if timeout_s is not None and (time.time() - t0) > float(timeout_s):
          self.shutdown()
          raise TimeoutError(
            f"No CameraInfo on any of "
            f"{[self.topics_for(r)['intrinsics'] for r in self.robot_ids]} "
            f"within {timeout_s}s.")
        if not logged:
          logger.info("Waiting for intrinsics to be published..")
          logged = True
        try:
          self._intrinsics_cond.wait(2)
        except KeyboardInterrupt:
          self.shutdown()
          raise

  # -- anchoring, exposed for the messaging service / visualizer ------------ #

  def is_anchored(self, robot_id) -> bool:
    s = self.streams.get(int(robot_id))
    return bool(s.anchored) if s else False

  def boot_enu(self, robot_id) -> np.ndarray:
    s = self.streams.get(int(robot_id))
    return np.zeros(3) if s is None else s.boot_enu.copy()

  def frames_robot(self, robot_id) -> int:
    s = self.streams.get(int(robot_id))
    return 0 if s is None else s.frames_yielded

  def domain_of(self, robot_id) -> int:
    for rid, did in zip(self.robot_ids, self.domain_ids):
      if rid == int(robot_id):
        return did
    return -1

  def local_to_world_shift(self, robot_id) -> np.ndarray:
    """The RDF translation that takes robot-local map coords into world."""
    return mrc.local_to_world_shift(self.boot_enu(robot_id),
                                    self.src_coord_system, "rdf")

  def world_to_local_shift(self, robot_id) -> np.ndarray:
    """The RDF translation that takes world coords back into robot-local."""
    return mrc.world_to_local_shift(self.boot_enu(robot_id),
                                    self.src_coord_system, "rdf")

  # -- iteration -------------------------------------------------------------- #

  def _next_msgs(self):
    """One round-robin sweep. Returns ``(stream, msgs)`` or ``None``."""
    ids = list(self.streams.keys())
    n = len(ids)
    for offset in range(n):
      rid = ids[(self._rr_index + offset) % n]
      s = self.streams[rid]
      try:
        msgs = s.msgs.get_nowait()
      except queue.Empty:
        continue
      self._rr_index = (self._rr_index + offset + 1) % n
      return s, msgs
    return None

  def __iter__(self):
    while True:
      # Clear BEFORE sweeping so a frame arriving during the sweep cannot be
      # missed (it would set the event again and we would not block).
      self._data_event.clear()
      item = self._next_msgs()
      if item is None:
        if self._shutdown_event.is_set():
          return
        self._data_event.wait(2.0)
        continue

      stream, msgs = item
      frame = self._decode(stream, msgs)
      if frame is None:
        continue
      stream.frames_yielded += 1
      self.frames_total += 1
      yield frame

  def _decode(self, stream, msgs):
    # Only swap channels when the message is ACTUALLY BGR-encoded.
    #
    # This used to swap unconditionally (the variable names bgra_img/bgr_img
    # are the fossil of that assumption, which holds for VOXL's bgr8/bgra8).
    # Isaac Sim publishes rgb8 — measured 2026-09-02 on
    # /robot_1/sensors/front_stereo/left/image_rect AND .../right/image_rect,
    # both `encoding: rgb8` — so the unconditional swap fed RADSeg red and
    # blue transposed on every frame of every run on this stack. Structural
    # queries (tree, debris pile, roof) still scored ~0.97 because their
    # features are shape-dominated, which is exactly why this hid for so long;
    # a colour-dominated query like `person` (skin, clothing) collapsed to a
    # 0.037 peak over 6039 voxels.
    #
    # This mirrors the known-good single-robot path in the reference tree
    # (datasets/ros.py: "Isaac Sim publishes rgb8 — flipping it would feed
    # RADIO swapped R/B."). Keep the two in sync.
    img = image_to_numpy(msgs["rgb"]).astype("float") / 255
    img = img[..., :3]
    if msgs["rgb"].encoding.lower().startswith("bgr"):
      img = img[..., (2, 1, 0)]
    rgb_img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)

    src_pose_4x4 = torch.tensor(
      pose_to_numpy(_RobotStream._pose_of(msgs["pose"])), dtype=torch.float)
    # local -> world, IN FLU, BEFORE rotating into RDF. Translation only.
    boot = stream.boot_enu
    src_pose_4x4[0, 3] += float(boot[0])
    src_pose_4x4[1, 3] += float(boot[1])
    src_pose_4x4[2, 3] += float(boot[2])   # 0.0 today; z stays AGL.
    rdf_pose_4x4 = g3d.transform_pose_4x4(src_pose_4x4, self.src2rdf_transform)

    depth_img = image_to_numpy(msgs["depth"])
    depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)

    if self.rgb_h != rgb_img.shape[-2] or self.rgb_w != rgb_img.shape[-1]:
      rgb_img = torch.nn.functional.interpolate(
        rgb_img.unsqueeze(0), size=(self.rgb_h, self.rgb_w),
        mode=self.interp_mode,
        antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

    if self.depth_h != depth_img.shape[-2] or \
        self.depth_w != depth_img.shape[-1]:
      depth_img = torch.nn.functional.interpolate(
        depth_img.unsqueeze(0), size=(self.depth_h, self.depth_w),
        mode="nearest-exact").squeeze(0)

    if torch.sum(~depth_img.isnan()) == 0:
      logger.warning("[%s] Ignoring depth frame with no valid values",
                     stream.robot_name)
      return None

    return dict(rgb_img=rgb_img, depth_img=depth_img, pose_4x4=rdf_pose_4x4,
                robot_id=int(stream.robot_id), robot_name=stream.robot_name)

  # -- teardown --------------------------------------------------------------- #

  def shutdown(self):
    self._shutdown_event.set()
    self._data_event.set()
    with self._intrinsics_cond:
      self._intrinsics_cond.notify_all()
    for s in self.streams.values():
      s.shutdown()
    logger.info("MultiRobotRos2Subscriber shutdown.")


def _normalize_offsets(offsets) -> Dict[str, List[float]]:
  """Accept ``{1: [x, y]}``, ``{"1": [x, y]}`` or a JSON string."""
  if offsets is None:
    return dict()
  if isinstance(offsets, str):
    offsets = json.loads(offsets)
  out = dict()
  for k, v in dict(offsets).items():
    out[str(k)] = [float(v[0]), float(v[1])]
  return out
