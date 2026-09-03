"""Fake robots and topic sniffers for the ``ros``-marked tests.

Each helper stands up a real rclpy participant on its own private context and
domain, so the tests exercise the same cross-context discovery the shared
server relies on rather than an in-process shortcut.

Domain ids used by the tests are deliberately odd (90+) so they cannot collide
with a live robot on 1..8 or the gossip layer on 99.
"""

import math
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from sensor_msgs.msg import Image, CameraInfo, NavSatFix, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import std_msgs.msg

try:
  from rclpy.signals import SignalHandlerOptions
except ImportError:  # pragma: no cover
  SignalHandlerOptions = None

RELIABLE_10 = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
BEST_EFFORT_1 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST, depth=1)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def make_context(domain_id):
  ctx = rclpy.Context()
  kwargs = dict(context=ctx, domain_id=int(domain_id))
  if SignalHandlerOptions is not None:
    kwargs["signal_handler_options"] = SignalHandlerOptions.NO
  rclpy.init(**kwargs)
  return ctx


class SpinningNode:
  """A node on its own private context, spun by its own thread."""

  def __init__(self, name, domain_id):
    self.context = make_context(domain_id)
    self.node = Node(name, context=self.context)
    self.executor = SingleThreadedExecutor(context=self.context)
    self.executor.add_node(self.node)
    self._thread = threading.Thread(target=self._spin, daemon=True)
    self._thread.start()

  def _spin(self):
    try:
      self.executor.spin()
    except Exception:
      pass

  def shutdown(self):
    try:
      self.executor.remove_node(self.node)
    except Exception:
      pass
    try:
      self.node.destroy_node()
    except Exception:
      pass
    try:
      self.context.try_shutdown()
    except Exception:
      pass


def _stamp(node, t):
  msg = std_msgs.msg.Header()
  msg.stamp.sec = int(t)
  msg.stamp.nanosec = int((t - int(t)) * 1e9)
  msg.frame_id = "map"
  return msg


class FakeRobot(SpinningNode):
  """Publishes everything MultiRobotRos2Subscriber subscribes to."""

  def __init__(self, robot_id, domain_id, width=16, height=16,
               fx=8.0, fy=8.0, use_pose_stamped=False):
    super().__init__(f"fake_robot_{robot_id}", domain_id)
    self.robot_id = robot_id
    self.name = f"robot_{robot_id}"
    self.width, self.height = width, height
    self.fx, self.fy = fx, fy

    n = self.node
    p = f"/{self.name}"
    self.info_pub = n.create_publisher(
      CameraInfo, f"{p}/sensors/front_stereo/left/camera_info", BEST_EFFORT_1)
    self.rgb_pub = n.create_publisher(
      Image, f"{p}/sensors/front_stereo/left/image_rect", RELIABLE_10)
    self.depth_pub = n.create_publisher(
      Image, f"{p}/sensors/front_stereo/left/depth_ground_truth", RELIABLE_10)
    if use_pose_stamped:
      self.pose_pub = n.create_publisher(
        PoseStamped, f"{p}/odometry_conversion/pose_stamped", RELIABLE_10)
      self._pose_stamped = True
    else:
      self.pose_pub = n.create_publisher(
        Odometry, f"{p}/odometry_conversion/odometry", RELIABLE_10)
      self._pose_stamped = False
    self.fix_pub = n.create_publisher(
      NavSatFix, f"{p}/interface/mavros/global_position/global", BEST_EFFORT_1)

  # -- messages ------------------------------------------------------------- #

  def camera_info(self):
    m = CameraInfo()
    m.header = _stamp(self.node, 0.0)
    m.width, m.height = self.width, self.height
    m.k = [self.fx, 0.0, self.width / 2.0,
           0.0, self.fy, self.height / 2.0,
           0.0, 0.0, 1.0]
    return m

  def rgb(self, t, value=128):
    arr = np.full((self.height, self.width, 3), value, dtype=np.uint8)
    m = Image()
    m.header = _stamp(self.node, t)
    m.height, m.width = self.height, self.width
    m.encoding = "rgb8"
    m.is_bigendian = 0
    m.step = self.width * 3
    m.data = arr.tobytes()
    return m

  def depth(self, t, value=5.0):
    arr = np.full((self.height, self.width), value, dtype=np.float32)
    m = Image()
    m.header = _stamp(self.node, t)
    m.height, m.width = self.height, self.width
    m.encoding = "32FC1"
    m.is_bigendian = 0
    m.step = self.width * 4
    m.data = arr.tobytes()
    return m

  def pose(self, t, xyz):
    if self._pose_stamped:
      m = PoseStamped()
      m.header = _stamp(self.node, t)
      m.pose.position.x, m.pose.position.y, m.pose.position.z = map(
        float, xyz)
      m.pose.orientation.w = 1.0
      return m
    m = Odometry()
    m.header = _stamp(self.node, t)
    m.child_frame_id = "base_link"
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = \
        map(float, xyz)
    m.pose.pose.orientation.w = 1.0
    return m

  def navsat(self, t, world_xy, origin=(38.736832, -9.137977, 90.0)):
    """A fix that, run through gps_to_enu, gives ``world_xy``."""
    lat0, lon0, alt0 = origin
    m = NavSatFix()
    m.header = _stamp(self.node, t)
    m.latitude = world_xy[1] / 111320.0 + lat0
    m.longitude = (world_xy[0] / (111320.0 * math.cos(math.radians(lat0)))
                   + lon0)
    m.altitude = alt0
    m.status.status = 0
    return m

  # -- publishing ------------------------------------------------------------ #

  def publish_info(self, n=5, period=0.05):
    for _ in range(n):
      self.info_pub.publish(self.camera_info())
      time.sleep(period)

  def publish_frame(self, t, odom_xyz, rgb_value=128, depth_value=5.0):
    self.rgb_pub.publish(self.rgb(t, rgb_value))
    self.depth_pub.publish(self.depth(t, depth_value))
    self.pose_pub.publish(self.pose(t, odom_xyz))

  def publish_fix(self, t, world_xy):
    self.fix_pub.publish(self.navsat(t, world_xy))


class Sniffer(SpinningNode):
  """Subscribes to a topic and records every message."""

  def __init__(self, name, domain_id, msg_type, topic, qos=RELIABLE_10):
    super().__init__(name, domain_id)
    self.messages = []
    self._lock = threading.Lock()
    self.sub = self.node.create_subscription(msg_type, topic, self._cb, qos)

  def _cb(self, msg):
    with self._lock:
      self.messages.append(msg)

  def wait_for(self, n=1, timeout=15.0, pump=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
      with self._lock:
        if len(self.messages) >= n:
          return list(self.messages)
      if pump is not None:
        pump()
      time.sleep(0.05)
    with self._lock:
      return list(self.messages)

  def count(self):
    with self._lock:
      return len(self.messages)

  def clear(self):
    with self._lock:
      self.messages.clear()


class Talker(SpinningNode):
  """Publishes std_msgs/String on a topic."""

  def __init__(self, name, domain_id, topic, qos=RELIABLE_10):
    super().__init__(name, domain_id)
    self.pub = self.node.create_publisher(std_msgs.msg.String, topic, qos)

  def say(self, text):
    m = std_msgs.msg.String()
    m.data = text
    self.pub.publish(m)


def wait_until(predicate, timeout=15.0, period=0.05):
  deadline = time.time() + timeout
  while time.time() < deadline:
    if predicate():
      return True
    time.sleep(period)
  return predicate()
