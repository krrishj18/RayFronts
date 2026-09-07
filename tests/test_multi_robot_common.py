"""Pure logic behind the shared server: query bookkeeping, naming, status.

Runs on a bare python. The ROS-level versions of these (real topics, real
messages) live in test_multi_messaging.py.
"""

import json

import pytest


# --------------------------------------------------------------------------- #
# Topic naming -- part of the contract with raven_nav, which parses labels back
# out of q{k}_{label} topic names.
# --------------------------------------------------------------------------- #

def _original_sanitize(s):
  """The body that used to live in Ros2MessagingService._sanitize_topic_name.

  Vendored verbatim so a refactor of the shared helper cannot silently change
  topic names under raven_nav.
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


@pytest.mark.parametrize("raw", [
  "person", "fallen tree", "wood planks", "  spaced  out  ", "a/b",
  "UPPER Case", "trailing-", "-leading", "", "  ", "under_score",
  "many   spaces", "punct!?.,", "mixed 1 2 3", "swimming pool",
])
def test_sanitize_matches_the_original(mrc, raw):
  assert mrc.sanitize_topic_name(raw) == _original_sanitize(raw)


def test_sanitize_never_produces_a_ros_illegal_segment(mrc):
  for raw in ["fallen tree", "a/b", "  x  ", "!!!", "9lives"]:
    out = mrc.sanitize_topic_name(raw)
    assert "__" not in out
    assert not out.startswith("_") and not out.endswith("_")


def test_query_topic_suffix(mrc):
  labels = ["person", "fallen tree", ""]
  assert mrc.query_topic_suffix(0, labels) == "q0_person"
  assert mrc.query_topic_suffix(1, labels) == "q1_fallen_tree"
  # Empty label -> bare index, and the segment never starts with a digit.
  assert mrc.query_topic_suffix(2, labels) == "q2"
  assert mrc.query_topic_suffix(7, None) == "q7"
  assert mrc.query_topic_suffix(9, labels) == "q9"


def test_visualization_topic_path_sanitizes_runtime_query_labels(mrc):
  raw = 'queries/["person", "utility pole"]/voxels'
  assert (mrc.sanitize_topic_path(raw)
          == "queries/person_utility_pole/voxels")
  assert mrc.sanitize_topic_path("queries/!!!/rays") == "queries/unnamed/rays"


def test_topic_templates(mrc):
  assert mrc.robot_name(2) == "robot_2"
  assert mrc.robot_name("robot_9") == "robot_9"
  assert (mrc.fill_topic_template("/{robot}/rayfronts/status", 3)
          == "/robot_3/rayfronts/status")
  assert (mrc.fill_topic_template("/{robot}/a/{id}", 4) == "/robot_4/a/4")
  assert mrc.fill_topic_template(None, 1) is None
  # No braces to substitute is fine, and a stray brace must not raise.
  assert mrc.fill_topic_template("/plain/topic", 1) == "/plain/topic"
  assert mrc.fill_topic_template("/{weird}/x", 1) == "/{weird}/x"


def test_frozen_output_topic_names(mrc):
  """The names agent A and agent C code against."""
  rid = 1
  prefix = mrc.fill_topic_template("/{robot}/rayfronts/msg_serv", rid)
  assert prefix == "/robot_1/rayfronts/msg_serv"
  assert f"{prefix}/voxels_sim/all" == "/robot_1/rayfronts/msg_serv/voxels_sim/all"
  assert f"{prefix}/rays_sim/all" == "/robot_1/rayfronts/msg_serv/rays_sim/all"
  assert f"{prefix}/frontiers" == "/robot_1/rayfronts/msg_serv/frontiers"
  assert (f"{prefix}/new_text_query"
          == "/robot_1/rayfronts/msg_serv/new_text_query")
  assert (f"{prefix}/guiding_queries"
          == "/robot_1/rayfronts/msg_serv/guiding_queries")
  assert (mrc.fill_topic_template("/{robot}/rayfronts/status", rid)
          == "/robot_1/rayfronts/status")
  assert (mrc.fill_topic_template("/{robot}/rayfronts", rid)
          == "/robot_1/rayfronts")


# --------------------------------------------------------------------------- #
# Guiding-query union / refcount / deletion
# --------------------------------------------------------------------------- #

def test_single_robot_add_and_drop(mrc):
  r = mrc.GuidingQueryRegistry()
  added, removed = r.set_guiding(1, ["mailbox", "car"])
  assert added == ["mailbox", "car"]
  assert removed == []
  assert r.union() == ["mailbox", "car"]

  added, removed = r.set_guiding(1, ["car"])
  assert added == []
  assert removed == ["mailbox"]
  assert r.union() == ["car"]


def test_a_label_two_robots_want_survives_one_dropping_it(mrc):
  r = mrc.GuidingQueryRegistry()
  r.set_guiding(1, ["car", "mailbox"])
  added, removed = r.set_guiding(2, ["car", "fence"])
  # "car" was already in the union, so it is not re-added.
  assert added == ["fence"]
  assert removed == []
  assert r.refcount("car") == 2

  added, removed = r.set_guiding(1, [])
  # robot_1 dropped both, but robot_2 still wants "car".
  assert added == []
  assert removed == ["mailbox"]
  assert r.refcount("car") == 1

  added, removed = r.set_guiding(2, [])
  assert removed == ["car", "fence"]
  assert r.union() == []


def test_pinned_labels_are_never_deleted(mrc):
  """new_text_query labels are the OG's target/background: exempt."""
  r = mrc.GuidingQueryRegistry()
  assert r.pin(["person", "road"]) == ["person", "road"]
  # The LVLM happens to also name "road" as a clue.
  added, _ = r.set_guiding(1, ["road", "mailbox"])
  assert added == ["mailbox"]          # "road" already known
  added, removed = r.set_guiding(1, [])
  assert removed == ["mailbox"]        # "road" is pinned -> not removed
  assert not r.deletable("road")
  assert not r.deletable("person")
  assert r.union() == ["person", "road"]


def test_pin_after_guiding_protects_it(mrc):
  r = mrc.GuidingQueryRegistry()
  r.set_guiding(1, ["car"])
  assert r.deletable("car") is False or r.refcount("car") == 1
  r.pin(["car"])
  _, removed = r.set_guiding(1, [])
  assert removed == []
  assert not r.deletable("car")


def test_duplicate_and_blank_labels_are_normalised(mrc):
  r = mrc.GuidingQueryRegistry()
  added, _ = r.set_guiding(1, ["car", "car", "  car ", "", None, " truck "])
  assert added == ["car", "truck"]
  assert r.guiding_of(1) == ["car", "truck"]


def test_drop_robot_releases_its_labels(mrc):
  r = mrc.GuidingQueryRegistry()
  r.set_guiding(1, ["a", "b"])
  r.set_guiding(2, ["b"])
  assert sorted(r.drop_robot(1)) == ["a"]
  assert r.refcount("b") == 1


def test_union_order_is_pinned_then_guiding(mrc):
  r = mrc.GuidingQueryRegistry()
  r.pin(["person"])
  r.set_guiding(2, ["z"])
  r.set_guiding(1, ["y"])
  # Pinned first (registration order), then guiding by robot id.
  assert r.union() == ["person", "y", "z"]


@pytest.mark.parametrize("payload,expected", [
  ('["car", "mailbox"]', ["car", "mailbox"]),
  ("[]", []),
  ("", []),
  (None, []),
  ('"car"', ["car"]),
  ("car, mailbox", ["car", "mailbox"]),          # human typing on the CLI
  ('{"guiding_objects": ["a", "b"]}', ["a", "b"]),
  ('["  padded  ", ""]', ["padded"]),
  ("not json at all", ["not json at all"]),
])
def test_parse_guiding_payload(mrc, payload, expected):
  assert mrc.parse_guiding_payload(payload) == expected


# --------------------------------------------------------------------------- #
# Status payload (FROZEN schema -- semantic_search_task reads it)
# --------------------------------------------------------------------------- #

def test_status_schema_is_exactly_the_frozen_keys(mrc):
  s = mrc.build_status(robot_id=2, domain_id=2, anchored=True,
                       boot_enu=[1.0, 2.0, 0.0], frames_robot=7,
                       frames_total=19, queries=["person", "road"],
                       vox_count=1234, ray_count=56, ts=1.5)
  assert list(s.keys()) == list(mrc.STATUS_KEYS)
  assert s["robot"] == "robot_2"
  assert s["domain"] == 2
  assert s["anchored"] is True
  assert s["boot_enu"] == [1.0, 2.0, 0.0]
  assert s["frames_robot"] == 7
  assert s["frames_total"] == 19
  assert s["queries"] == ["person", "road"]
  assert s["vox_count"] == 1234
  assert s["ray_count"] == 56
  assert s["ts"] == 1.5


def test_status_round_trips_through_json(mrc):
  s = mrc.build_status(1, 1, False, None, 0, 0, None, 0, 0, 0.0)
  back = json.loads(mrc.status_to_json(s))
  assert back == s
  assert back["boot_enu"] is None
  assert back["queries"] == []


def test_status_types_are_json_safe(mrc):
  import numpy as np
  s = mrc.build_status(np.int64(3), np.int64(3), np.bool_(True),
                       np.array([1.0, 2.0, 3.0]), np.int64(4), np.int64(9),
                       ("person",), np.int64(10), np.int64(11),
                       np.float64(2.5))
  json.dumps(s)  # must not raise
  assert isinstance(s["domain"], int)
  assert isinstance(s["anchored"], bool)
  assert all(isinstance(v, float) for v in s["boot_enu"])


def test_status_gate_predicate_semantics(mrc):
  """What semantic_search_task's shared mode will evaluate."""
  def ready(status, required):
    return bool(status["anchored"]) and status["frames_robot"] >= required

  assert not ready(mrc.build_status(1, 1, False, None, 99, 99, [], 0, 0, 0), 5)
  assert not ready(mrc.build_status(1, 1, True, [0, 0, 0], 4, 4, [], 0, 0, 0), 5)
  assert ready(mrc.build_status(1, 1, True, [0, 0, 0], 5, 5, [], 0, 0, 0), 5)
