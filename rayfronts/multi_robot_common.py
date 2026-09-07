"""Torch-free helpers shared by the multi-robot RayFronts server.

Everything in this module is deliberately restricted to the standard library
(plus numpy) and imports **nothing** from torch, rclpy, hydra or the rest of the
``rayfronts`` package.  That makes it importable — and unit testable — on a bare
python interpreter, which is where most of the multi-robot logic (frame shifts,
guiding-query bookkeeping, topic naming, status payloads) is actually tested.

Because ``rayfronts/__init__.py`` imports the heavy sub-packages, tests that run
outside the robot container load this file directly with
``importlib.util.spec_from_file_location`` rather than ``import rayfronts...``.
Keep it dependency-free.

Contents:
  * coordinate-system helpers mirroring ``rayfronts.geometry3d`` without torch,
  * the local(robot ``map``) <-> world(shared ENU) shift used by the shared map,
  * the flat-earth ``gps_to_enu`` used to anchor a robot from its NavSatFix,
  * ROS topic-name sanitation shared with ``Ros2MessagingService``,
  * :class:`GuidingQueryRegistry` — the union/refcount rule for LVLM guiding
    queries coming from several robots,
  * the frozen status-topic payload builder,
  * the OPEN-SET PROBE inputs (``RAYFRONTS_PROBE_PROMPTS`` /
    ``RAYFRONTS_PROBE_POINTS``) and the nearest-voxel selection they drive.
"""

import json
import math
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Coordinate systems
# --------------------------------------------------------------------------- #

# Same table as rayfronts.geometry3d.get_coord_system_transform.
_AXES = dict(r=0, l=0, u=1, d=1, f=2, b=2)


def coord_system_transform(src: str, tgt: str) -> np.ndarray:
  """Numpy port of ``rayfronts.geometry3d.get_coord_system_transform``.

  Args:
    src: length-3 string naming the source convention, e.g. ``"flu"``.
    tgt: length-3 string naming the target convention, e.g. ``"rdf"``.

  Returns:
    A (3, 3) float64 change-of-basis matrix ``T`` such that
    ``p_tgt = T @ p_src``.

  ``tests/test_frame_shift.py`` asserts this matches the torch implementation
  element-for-element for every convention pair we care about.
  """
  if len(src) != 3 or len(tgt) != 3:
    raise ValueError(f"Coordinate systems must be 3 letters, got {src}/{tgt}")
  t = np.zeros((3, 3), dtype=np.float64)
  for i, tgt_dir in enumerate(tgt.lower()):
    a = _AXES[tgt_dir]
    for j, src_dir in enumerate(src.lower()):
      b = _AXES[src_dir]
      if a == b:
        t[i, j] = 1.0 if src_dir == tgt_dir else -1.0
        break
  return t


def transform_offset(offset_xyz: Sequence[float],
                     src: str = "flu",
                     tgt: str = "rdf") -> np.ndarray:
  """Express a *translation* given in ``src`` axes in ``tgt`` axes.

  A pure translation transforms as a vector: ``d_tgt = T_src2tgt @ d_src``.
  """
  t = coord_system_transform(src, tgt)
  d = np.asarray(offset_xyz, dtype=np.float64).reshape(3)
  return t @ d


def world_flu_to_rdf(xyz: Sequence[float]) -> np.ndarray:
  """World FLU ``(x, y, z)`` -> map RDF ``(-y, -z, x)``.

  Everything the mapper stores (``global_vox_xyz``, ``ray_orig_angles``) is RDF;
  every number a human types (a GT casualty position, a search-area corner) is
  world FLU.  A 2-element point is accepted and padded with ``z = 0``.

  This is the exact inverse of the ``px, py, pz = (xyz[2], -xyz[0], -xyz[1])``
  line the ``[raw top]`` logger uses to print a voxel in world coordinates.
  """
  p = [float(v) for v in list(xyz)[:3]]
  while len(p) < 3:
    p.append(0.0)
  return transform_offset(p, "flu", "rdf")


def rdf_to_world_flu(xyz: Sequence[float]) -> np.ndarray:
  """Map RDF ``(x, y, z)`` -> world FLU ``(z, -x, -y)``."""
  p = [float(v) for v in list(xyz)[:3]]
  while len(p) < 3:
    p.append(0.0)
  return transform_offset(p, "rdf", "flu")


def local_to_world_shift(boot_enu: Sequence[float],
                         src_coord_system: str = "flu",
                         tgt_coord_system: str = "rdf") -> np.ndarray:
  """Shift that turns robot-local coordinates into shared-world coordinates.

  ``boot_enu`` is the robot's ``map``-frame origin expressed in the shared world
  ENU frame — i.e. ``world_flu = local_flu + boot_enu`` (this is exactly
  ``raven_nav._local_to_world``).  Everything downstream of the dataset lives in
  RDF, so the shift the mapper sees is the same vector re-expressed in RDF.

  With ``flu -> rdf`` (``T = [[0,-1,0],[0,0,-1],[1,0,0]]``) the RDF delta for a
  world FLU offset ``(bx, by, bz)`` is ``(-by, -bz, bx)``.  Since we only ever
  anchor in xy (``bz`` is forced to 0, z stays AGL) that reduces to
  ``(-by, 0, bx)``.
  """
  return transform_offset(boot_enu, src_coord_system, tgt_coord_system)


def world_to_local_shift(boot_enu: Sequence[float],
                         src_coord_system: str = "flu",
                         tgt_coord_system: str = "rdf") -> np.ndarray:
  """Inverse of :func:`local_to_world_shift` — used when publishing back.

  Map outputs live in the shared world; robot *i* expects them in its own
  ``map`` frame, so we add ``-T @ boot_enu`` = ``(by, bz, -bx)`` in RDF.
  """
  return -local_to_world_shift(boot_enu, src_coord_system, tgt_coord_system)


# --------------------------------------------------------------------------- #
# GPS -> ENU (10-line re-implementation of coordination_bringup.frame_utils)
# --------------------------------------------------------------------------- #

# Keep these in sync with
# common/ros_packages/coordination/coordination_bringup/coordination_bringup/
#   frame_utils.py  (and gcs_utils.py, simulation/.../gps_utils.py — all three
# use the same "Lisbon" origin, which is what makes every robot's boot_enu
# comparable in one world frame).
DEFAULT_ORIGIN_LAT = 38.736832
DEFAULT_ORIGIN_LON = -9.137977
DEFAULT_ORIGIN_ALT = 90.0


def gps_to_enu(lat: float, lon: float, alt: float,
               origin_lat: float = DEFAULT_ORIGIN_LAT,
               origin_lon: float = DEFAULT_ORIGIN_LON,
               origin_alt: float = DEFAULT_ORIGIN_ALT
               ) -> Tuple[float, float, float]:
  """Flat-earth GPS -> ENU metres, byte-identical to ``frame_utils.gps_to_enu``."""
  x = (lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
  y = (lat - origin_lat) * 111320.0
  z = alt - origin_alt
  return x, y, z


# --------------------------------------------------------------------------- #
# Topic naming
# --------------------------------------------------------------------------- #

def robot_name(robot_id) -> str:
  """``1 -> "robot_1"``.  A string id is passed through untouched."""
  if isinstance(robot_id, str):
    return robot_id
  return f"robot_{int(robot_id)}"


def fill_topic_template(template: str, robot_id) -> str:
  """Substitute ``{robot}`` / ``{id}`` in a topic template.

  ``str.replace`` rather than ``str.format`` so a template containing any other
  brace (there should not be one, but a config is a config) cannot raise.
  """
  if template is None:
    return None
  name = robot_name(robot_id)
  ident = str(robot_id)
  return template.replace("{robot}", name).replace("{id}", ident)


def sanitize_topic_name(s: str) -> str:
  """Make a string safe for ROS 2 topic names (alphanumeric and underscore).

  Verbatim port of ``Ros2MessagingService._sanitize_topic_name`` — that method
  now delegates here so the shared server and the legacy per-robot server can
  never drift.  ``raven_nav._detect_rayfronts_labels`` parses the result back
  out of ``q{k}_{label}`` topic names, so this function is part of the
  cross-package contract.
  """
  if not isinstance(s, str) or not s:
    return ""
  out = []
  for c in s:
    if c.isalnum() or c == "_":
      out.append(c)
    elif c.isspace() or not c.isalnum():
      out.append("_")
  name = "".join(out)
  while "__" in name:
    name = name.replace("__", "_")
  return name.strip("_") or ""


def sanitize_topic_path(path: str) -> str:
  """Sanitize every segment of a relative ROS topic path.

  Visualization layers may retain ``/`` hierarchy, but query labels are
  runtime strings and can contain JSON punctuation, quotes, or brackets.
  Passing those characters directly to ``create_publisher`` terminates the
  shared mapper. Empty/fully-punctuation segments get a stable placeholder.
  """
  segments = str(path).split("/")
  return "/".join(sanitize_topic_name(segment) or "unnamed"
                  for segment in segments)


def query_topic_suffix(q: int, query_labels: Optional[Sequence] = None) -> str:
  """``q{index}_{sanitized label}`` (or ``q{index}`` when there is no label)."""
  if query_labels is not None and q < len(query_labels):
    sanitized = sanitize_topic_name(str(query_labels[q]))
    if sanitized:
      return f"q{q}_{sanitized}"
  return f"q{q}"


# --------------------------------------------------------------------------- #
# Guiding-query bookkeeping
# --------------------------------------------------------------------------- #

class GuidingQueryRegistry:
  """Union + refcount over the per-robot LVLM guiding-query lists.

  The shared mapper has ONE query column set.  Two kinds of label reach it:

  * ``new_text_query`` labels (the mission's target/background vocabulary).
    These are **pinned**: they are never removed.  This mirrors the original
    RAVEN ``delete_queries``, which exempts ``_target_objects`` and
    ``_background_objects``.
  * ``guiding_queries`` labels — each robot publishes its *current* guiding list
    (the LVLM's latest answer).  A guiding label survives while at least one
    robot still lists it; when the last robot drops it, it is deleted.

  The registry only decides membership; the column order and the feature matrix
  are owned by the mapping server (registration order, exactly as today).
  """

  def __init__(self):
    self._lock = threading.RLock()
    self._pinned: List[str] = []          # order preserved, deduped
    self._pinned_set = set()
    self._per_robot: Dict[object, List[str]] = {}

  # -- pinned (new_text_query) --------------------------------------------- #

  def pin(self, labels) -> List[str]:
    """Pin one or more labels.  Returns the labels that were not known before.

    "Not known before" means not pinned AND not already present as a guiding
    label — the caller uses the return value to decide what to encode, and the
    mapping server's own ``_queries_labels_history`` de-dupes anyway.
    """
    if isinstance(labels, str):
      labels = [labels]
    fresh = []
    with self._lock:
      known = self._known_locked()
      for label in labels:
        label = _norm_label(label)
        if not label:
          continue
        if label not in self._pinned_set:
          self._pinned.append(label)
          self._pinned_set.add(label)
        if label not in known:
          fresh.append(label)
          known.add(label)
    return fresh

  def is_pinned(self, label: str) -> bool:
    with self._lock:
      return _norm_label(label) in self._pinned_set

  @property
  def pinned(self) -> List[str]:
    with self._lock:
      return list(self._pinned)

  # -- guiding (per robot) -------------------------------------------------- #

  def set_guiding(self, robot_id, labels) -> Tuple[List[str], List[str]]:
    """Replace robot ``robot_id``'s guiding list.

    Returns:
      ``(added, removed)`` where ``added`` are labels that are new to the union
      (so the server must encode and append them as query columns) and
      ``removed`` are guiding labels no robot lists any more and that are not
      pinned (so the server must drop those columns).  Both lists preserve a
      deterministic order: ``added`` in the order given, ``removed`` in the
      order they were previously held by this robot.
    """
    if isinstance(labels, str):
      labels = [labels]
    new_list: List[str] = []
    seen = set()
    for label in labels or []:
      label = _norm_label(label)
      if label and label not in seen:
        new_list.append(label)
        seen.add(label)

    with self._lock:
      before = self._known_locked()
      old_list = self._per_robot.get(robot_id, [])
      self._per_robot[robot_id] = new_list
      after = self._known_locked()

      added = [l for l in new_list if l not in before]
      removed = [l for l in old_list if l not in after]
    return added, removed

  def guiding_of(self, robot_id) -> List[str]:
    with self._lock:
      return list(self._per_robot.get(robot_id, []))

  def drop_robot(self, robot_id) -> List[str]:
    """Forget a robot entirely; returns the labels that became unreferenced."""
    with self._lock:
      old_list = self._per_robot.pop(robot_id, [])
      after = self._known_locked()
      return [l for l in old_list if l not in after]

  def refcount(self, label: str) -> int:
    """How many robots currently list ``label`` as a guiding query."""
    label = _norm_label(label)
    with self._lock:
      return sum(1 for v in self._per_robot.values() if label in v)

  def union(self) -> List[str]:
    """Pinned labels first (registration order), then guiding labels."""
    with self._lock:
      out = list(self._pinned)
      seen = set(self._pinned_set)
      for robot_id in sorted(self._per_robot, key=_sort_key):
        for label in self._per_robot[robot_id]:
          if label not in seen:
            out.append(label)
            seen.add(label)
      return out

  def deletable(self, label: str) -> bool:
    """True when ``label`` may be removed from the shared query set."""
    label = _norm_label(label)
    with self._lock:
      if label in self._pinned_set:
        return False
      return self.refcount(label) == 0

  # -- internals ------------------------------------------------------------ #

  def _known_locked(self) -> set:
    known = set(self._pinned_set)
    for v in self._per_robot.values():
      known.update(v)
    return known


def _norm_label(label) -> str:
  if label is None:
    return ""
  return str(label).strip()


def _sort_key(x):
  try:
    return (0, float(x), "")
  except (TypeError, ValueError):
    return (1, 0.0, str(x))


def parse_guiding_payload(data: str) -> List[str]:
  """Parse the ``guiding_queries`` String payload into a list of labels.

  raven publishes a JSON list.  A bare comma-separated string is accepted too so
  a human can ``ros2 topic pub`` the topic by hand while debugging.
  """
  if data is None:
    return []
  data = data.strip()
  if not data:
    return []
  try:
    parsed = json.loads(data)
  except (ValueError, TypeError):
    return [x.strip() for x in data.split(",") if x.strip()]
  if isinstance(parsed, str):
    return [parsed.strip()] if parsed.strip() else []
  if isinstance(parsed, dict):
    parsed = parsed.get("guiding_objects", parsed.get("objects", []))
  if not isinstance(parsed, (list, tuple)):
    return []
  return [str(x).strip() for x in parsed if str(x).strip()]


# --------------------------------------------------------------------------- #
# Open-set probes (RAYFRONTS_PROBE_PROMPTS / RAYFRONTS_PROBE_POINTS)
# --------------------------------------------------------------------------- #
#
# A probe is a DIAGNOSTIC, never part of the mission.  Probe PROMPTS are texts
# we want the encoder's opinion on ("person lying down", "casualty", ...)
# without adding them as query columns — with querying.compute_prob=True every
# extra column moves the softmax the planner thresholds, so trialling a wording
# by adding it to the vocabulary would silently change the mission's behaviour.
# Probe POINTS are known world-FLU ground-truth positions (the placed
# casualties); logging what the map holds THERE is the only way to separate
# "the encoder does not recognise this body" from "the planner never looked".
#
# Both parsers are total: anything unparseable yields an empty list rather than
# an exception, because these arrive from a mission yaml through a container
# env var and must never be able to stop a mapping server from booting.

# RDF is (right, down, forward).  A probe point given as [x, y] (world FLU)
# constrains no height, so it is matched on the HORIZONTAL pair only.
RDF_AXES = (0, 1, 2)
RDF_HORIZONTAL_AXES = (0, 2)


def _dedupe_strs(items: Iterable) -> List[str]:
  out: List[str] = []
  seen = set()
  for x in items:
    s = "" if x is None else str(x).strip()
    if not s or s in seen:
      continue
    seen.add(s)
    out.append(s)
  return out


def parse_probe_prompts(spec) -> List[str]:
  """Parse ``RAYFRONTS_PROBE_PROMPTS`` into an ordered, deduped text list.

  Accepts a comma-separated string (the documented form), a JSON list, or an
  already-sequence value (a hydra ``ListConfig``, a python list).  Blank
  entries are dropped; order is the order given, because the probe scores are
  logged in that order and an operator reads them positionally.
  """
  if spec is None:
    return []
  if isinstance(spec, (bytes, bytearray)):
    spec = spec.decode("utf-8", "replace")
  if not isinstance(spec, str):
    try:
      items = list(spec)
    except TypeError:
      items = [spec]
    return _dedupe_strs(items)

  s = spec.strip()
  if not s:
    return []
  if s[0] in "[(":
    try:
      parsed = json.loads(s)
    except (ValueError, TypeError):
      parsed = None
    if isinstance(parsed, (list, tuple)):
      return _dedupe_strs(parsed)
    if isinstance(parsed, str):
      s = parsed
  return _dedupe_strs(s.split(","))


def _coerce_point(entry) -> Optional[Tuple[float, ...]]:
  """One probe point -> a 2- or 3-tuple of floats, or None if unusable.

  The ARITY is preserved on purpose: it is what decides whether the
  nearest-voxel search is horizontal (``[x, y]``) or full 3D (``[x, y, z]``).
  """
  if entry is None:
    return None
  if isinstance(entry, dict):
    if "x" not in entry or "y" not in entry:
      return None
    entry = ([entry["x"], entry["y"], entry["z"]] if "z" in entry
             else [entry["x"], entry["y"]])
  if isinstance(entry, (str, bytes, bytearray)):
    entry = str(entry).replace("[", " ").replace("]", " ").split(",")
  try:
    vals = [float(v) for v in list(entry)]
  except (TypeError, ValueError):
    return None
  if len(vals) < 2 or any(not math.isfinite(v) for v in vals):
    return None
  return tuple(vals[:3]) if len(vals) >= 3 else (vals[0], vals[1])


def parse_probe_points(spec) -> List[Tuple[float, ...]]:
  """Parse ``RAYFRONTS_PROBE_POINTS`` into world-FLU points of interest.

  The documented form is a JSON list of ``[x, y]`` (or ``[x, y, z]``) pairs::

      RAYFRONTS_PROBE_POINTS='[[-12.87,-33.52],[-15.06,-35.4]]'

  A single bare point (``[1, 2]``), a ``x,y;x,y`` string typed by hand, and a
  list of ``{"x":..,"y":..}`` dicts are all accepted too.  Anything else — a
  truncated JSON blob, a stray word, an entry with one number — is dropped
  silently; the caller logs how many points survived.
  """
  if spec is None:
    return []
  if isinstance(spec, (bytes, bytearray)):
    spec = spec.decode("utf-8", "replace")

  if isinstance(spec, str):
    s = spec.strip()
    if not s:
      return []
    try:
      raw = json.loads(s)
    except (ValueError, TypeError):
      # Not JSON: accept "x,y;x,y" / one point per line.
      raw = [chunk for chunk in s.replace(";", "\n").splitlines()
             if chunk.strip()]
  else:
    raw = spec

  if isinstance(raw, dict):
    raw = [raw]
  if raw is None:
    return []
  try:
    entries = list(raw)
  except TypeError:
    return []
  # A single flat point ([x, y] rather than [[x, y]]).
  if entries and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                     for v in entries):
    entries = [entries]

  out: List[Tuple[float, ...]] = []
  for entry in entries:
    p = _coerce_point(entry)
    if p is not None:
      out.append(p)
  return out


def probe_axes(point: Sequence[float]) -> Tuple[int, ...]:
  """Which RDF columns a probe point constrains.

  ``[x, y]`` says nothing about height, so it is matched on RDF columns
  ``(0, 2)`` = (right, forward); ``[x, y, z]`` is matched in full 3D.
  """
  return RDF_AXES if len(tuple(point)) >= 3 else RDF_HORIZONTAL_AXES


def nearest_index(points,
                  target: Sequence[float],
                  max_dist: Optional[float] = None,
                  axes: Optional[Sequence[int]] = None
                  ) -> Tuple[Optional[int], float]:
  """Row of ``points`` (Nx3, RDF) nearest ``target`` (3, RDF).

  Args:
    points: Nx3 array-like of RDF positions (``mapper.global_vox_xyz``).
    target: length-3 RDF position (run the FLU point through
      :func:`world_flu_to_rdf` first).
    max_dist: if given and the nearest row is further than this, the index is
      returned as ``None`` — the distance is still returned so the caller can
      say HOW far the nearest thing was.
    axes: columns to measure over; defaults to all three.

  Returns:
    ``(index_or_None, distance)``.  An empty/ill-shaped ``points`` gives
    ``(None, inf)``.
  """
  pts = np.asarray(points, dtype=np.float64)
  if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
    return None, float("inf")
  cols = list(RDF_AXES if axes is None else tuple(int(a) for a in axes))
  tgt = np.asarray(target, dtype=np.float64).reshape(-1)
  if tgt.shape[0] < 3:
    tgt = np.concatenate([tgt, np.zeros(3 - tgt.shape[0])])
  d = np.linalg.norm(pts[:, cols] - tgt[cols], axis=1)
  k = int(np.argmin(d))
  dist = float(d[k])
  if max_dist is not None and dist > float(max_dist):
    return None, dist
  return k, dist


# --------------------------------------------------------------------------- #
# Status topic
# --------------------------------------------------------------------------- #

# FROZEN (plan section 2.2).  semantic_search_task gates the mission on
# "anchored" and "frames_robot"; do not rename or drop a key.
STATUS_KEYS = ("robot", "domain", "anchored", "boot_enu", "frames_robot",
               "frames_total", "queries", "vox_count", "ray_count", "ts")


def build_status(robot_id,
                 domain_id: int,
                 anchored: bool,
                 boot_enu: Optional[Sequence[float]],
                 frames_robot: int,
                 frames_total: int,
                 queries: Optional[Iterable[str]],
                 vox_count: int,
                 ray_count: int,
                 ts: float) -> dict:
  """Build the ``/robot_i/rayfronts/status`` payload (frozen schema)."""
  return {
    "robot": robot_name(robot_id),
    "domain": int(domain_id),
    "anchored": bool(anchored),
    "boot_enu": (None if boot_enu is None
                 else [float(v) for v in list(boot_enu)[:3]]),
    "frames_robot": int(frames_robot),
    "frames_total": int(frames_total),
    "queries": [] if queries is None else [str(q) for q in queries],
    "vox_count": int(vox_count),
    "ray_count": int(ray_count),
    "ts": float(ts),
  }


def status_to_json(status: dict) -> str:
  return json.dumps(status, sort_keys=False)
