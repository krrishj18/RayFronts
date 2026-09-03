"""MultiRobotMappingServer's query bookkeeping, status, and open-set probes.

The mapper and the encoder are far too heavy to stand up in a unit test, so
these drive the parts that do not need them: the port of RAVEN's
``delete_queries``, the union/pin rules on top of it, the status payload, and
the PROBE path (whose one non-negotiable property — probe prompts never enter
the published query set — is checked here against a mocked mapper).
The full loop is exercised by the live run, not here.
"""

import logging
import math
import re
import threading
import types

import pytest

from conftest import HAVE_TORCH, ensure_pythonpath

pytestmark = [pytest.mark.torch,
              pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")]

if HAVE_TORCH:
  ensure_pythonpath()
  import torch
  from rayfronts import multi_robot_common as mrc
  from rayfronts.multi_robot_mapping_server import MultiRobotMappingServer


class _FakeDataset:
  def __init__(self):
    self.robot_ids = [1, 2]
    self.domain_ids = [1, 2]
    self.frames_total = 11
    self._frames = {1: 7, 2: 4}
    self._anchored = {1: True, 2: False}
    self._boot = {1: [0.0, 0.0, 0.0], 2: [100.0, 50.0, 0.0]}

  def is_anchored(self, rid):
    return self._anchored[int(rid)]

  def boot_enu(self, rid):
    return self._boot[int(rid)]

  def frames_robot(self, rid):
    return self._frames[int(rid)]

  def domain_of(self, rid):
    return int(rid)


class _FakeMessaging:
  def __init__(self):
    self.published = []

  def publish_status(self, rid, status):
    self.published.append((rid, status))


class _FakeMapper:
  def __init__(self, n_vox=13, n_ray=5):
    self.global_vox_xyz = torch.zeros(n_vox, 3)
    self.global_rays_orig_angles = torch.zeros(n_ray, 5)


def _bare_server(labels, feats=None):
  """A MultiRobotMappingServer with only the query state populated."""
  s = MultiRobotMappingServer.__new__(MultiRobotMappingServer)
  s._query_lock = threading.RLock()
  s._status_lock = threading.RLock()
  s.registry = mrc.GuidingQueryRegistry()
  s._queries_labels = dict(text=list(labels), img=[])
  if feats is None:
    feats = torch.arange(len(labels) * 4, dtype=torch.float).reshape(
      len(labels), 4)
  s._queries_feats = dict(text=feats, img=None)
  s._queries_labels_history = set(labels)
  s._queries_updated = False
  return s


# --------------------------------------------------------------------------- #
# delete_queries (port of the OG)
# --------------------------------------------------------------------------- #

def test_delete_removes_the_label_and_its_feature_row():
  s = _bare_server(["person", "car", "mailbox"])
  s.registry.pin(["person"])
  s.registry.set_guiding(1, ["car", "mailbox"])
  before = s._queries_feats["text"].clone()

  s.registry.set_guiding(1, ["car"])          # robot dropped "mailbox"
  s.delete_queries(["mailbox"])

  assert s._queries_labels["text"] == ["person", "car"]
  assert s._queries_feats["text"].shape == (2, 4)
  torch.testing.assert_close(s._queries_feats["text"], before[:2])
  # It leaves the history so the LVLM can name it again later.
  assert "mailbox" not in s._queries_labels_history
  assert s._queries_updated is True


def test_delete_keeps_column_order_of_the_survivors():
  s = _bare_server(["a", "b", "c", "d"])
  s.registry.set_guiding(1, ["a", "b", "c", "d"])
  s.registry.set_guiding(1, ["a", "c"])
  s.delete_queries(["b", "d"])
  assert s._queries_labels["text"] == ["a", "c"]
  torch.testing.assert_close(s._queries_feats["text"],
                             torch.tensor([[0., 1., 2., 3.],
                                           [8., 9., 10., 11.]]))


def test_pinned_labels_survive_a_delete_request():
  """new_text_query labels are the OG's target/background: never deleted."""
  s = _bare_server(["person", "road", "car"])
  s.registry.pin(["person", "road"])
  s.registry.set_guiding(1, ["car"])
  s.registry.set_guiding(1, [])
  s.delete_queries(["person", "road", "car"])
  assert s._queries_labels["text"] == ["person", "road"]
  assert s._queries_feats["text"].shape == (2, 4)


def test_a_label_another_robot_still_wants_is_not_deleted():
  s = _bare_server(["car", "fence"])
  s.registry.set_guiding(1, ["car", "fence"])
  s.registry.set_guiding(2, ["car"])
  s.registry.set_guiding(1, [])
  s.delete_queries(["car", "fence"])
  assert s._queries_labels["text"] == ["car"]


def test_delete_is_a_no_op_when_nothing_matches():
  s = _bare_server(["person"])
  s.registry.pin(["person"])
  s.delete_queries(["nonexistent"])
  assert s._queries_labels["text"] == ["person"]
  assert s._queries_updated is False


def test_delete_on_an_empty_query_set_does_not_explode():
  s = MultiRobotMappingServer.__new__(MultiRobotMappingServer)
  s._query_lock = threading.RLock()
  s.registry = mrc.GuidingQueryRegistry()
  s._queries_labels = None
  s._queries_feats = None
  s.delete_queries(["anything"])
  s._queries_labels = dict(text=[], img=[])
  s._queries_feats = dict(text=None, img=None)
  s.delete_queries(["anything"])


def test_delete_accepts_a_bare_string():
  s = _bare_server(["car"])
  s.registry.set_guiding(1, ["car"])
  s.registry.set_guiding(1, [])
  s.delete_queries("car")
  assert s._queries_labels["text"] == []


# --------------------------------------------------------------------------- #
# The guiding callback wiring
# --------------------------------------------------------------------------- #

def test_guiding_callback_adds_then_removes():
  s = _bare_server([])
  added, removed = [], []
  s.add_queries = lambda labels: added.append(list(labels))
  s.delete_queries = lambda labels: removed.append(list(labels))

  s._on_guiding_queries(1, ["car", "mailbox"])
  assert added == [["car", "mailbox"]]
  assert removed == []

  # robot_2 names one of the same objects: no duplicate column.
  s._on_guiding_queries(2, ["car"])
  assert added == [["car", "mailbox"]]

  # robot_1 changes its mind: "mailbox" is unreferenced, "car" is not.
  s._on_guiding_queries(1, ["fence"])
  assert added[-1] == ["fence"]
  assert removed == [["mailbox"]]


def test_new_text_query_pins_before_adding():
  s = _bare_server([])
  added = []
  s.add_queries = lambda labels: added.append(list(labels))
  s._on_new_text_query("person")
  assert added == [["person"]]
  assert s.registry.is_pinned("person")
  # A guiding list naming it later cannot make it deletable.
  s._on_guiding_queries(1, ["person"])
  s._on_guiding_queries(1, [])
  assert not s.registry.deletable("person")


def test_new_text_query_accepts_a_list_too():
  s = _bare_server([])
  added = []
  s.add_queries = lambda labels: added.append(list(labels))
  s._on_new_text_query(["person", "dog"])
  assert added == [["person", "dog"]]
  assert s.registry.is_pinned("dog")


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

def test_current_query_labels_is_column_order():
  s = _bare_server(["person", "car"])
  s._queries_labels["img"] = ["/tmp/a.png"]
  assert s.current_query_labels() == ["person", "car", "/tmp/a.png"]


def test_publish_status_once_reports_every_robot():
  s = _bare_server(["person", "car"])
  s.dataset = _FakeDataset()
  s.messaging_service = _FakeMessaging()
  s.mapper = _FakeMapper(n_vox=13, n_ray=5)
  s.publish_status_once()

  published = dict(s.messaging_service.published)
  assert set(published) == {1, 2}
  p1, p2 = published[1], published[2]
  assert list(p1.keys()) == list(mrc.STATUS_KEYS)
  assert p1["robot"] == "robot_1" and p1["anchored"] is True
  assert p1["boot_enu"] == [0.0, 0.0, 0.0]
  assert p1["frames_robot"] == 7 and p1["frames_total"] == 11
  assert p1["queries"] == ["person", "car"]
  assert p1["vox_count"] == 13 and p1["ray_count"] == 5
  # Unanchored robots report boot_enu null so the gate cannot be fooled.
  assert p2["anchored"] is False and p2["boot_enu"] is None


def test_status_survives_an_empty_map():
  s = _bare_server([])
  s.dataset = _FakeDataset()
  s.messaging_service = _FakeMessaging()
  s.mapper = _FakeMapper()
  s.mapper.global_vox_xyz = None
  s.mapper.global_rays_orig_angles = None
  s.publish_status_once()
  assert dict(s.messaging_service.published)[1]["vox_count"] == 0
  assert dict(s.messaging_service.published)[1]["ray_count"] == 0


# --------------------------------------------------------------------------- #
# Open-set probes
# --------------------------------------------------------------------------- #

MISSION_PROMPTS = ("person lying down,person lying on the ground,casualty,"
                   "human body,mannequin")
MISSION_POINTS = '[[-12.87,-33.52],[-15.06,-35.4],[-4.79,-69.42]]'


class _FakeEncoder:
  """Counts which encode_* path was taken and hands back unit-ish vectors."""

  def __init__(self, dim=4):
    self.dim = dim
    self.label_calls = []
    self.prompt_calls = []

  def _feats(self, texts):
    n = len(texts)
    f = torch.zeros(n, self.dim)
    for i in range(n):
      f[i, i % self.dim] = 1.0
      f[i, (i + 1) % self.dim] = 0.25
    return f

  def encode_labels(self, texts):
    self.label_calls.append(list(texts))
    return self._feats(texts)

  def encode_prompts(self, texts):
    self.prompt_calls.append(list(texts))
    return self._feats(texts)


class _ProbeMapper:
  """A mapper whose ``feature_query`` is a real cosine over fixed voxel feats.

  Mirrors SemanticRayFrontiersMap: ``vox_sim`` is [Q, N] (compute_cos_sim
  returns [N, Q] and the mapper transposes) and indexes ``vox_xyz`` row-wise.
  """

  def __init__(self, vox_xyz, vox_feat):
    self.global_vox_xyz = vox_xyz
    self.global_rays_orig_angles = None
    self._vox_feat = vox_feat
    self.query_rows = []

  def feature_query(self, feat_query, softmax=False, compressed=True):
    self.query_rows.append(int(feat_query.shape[0]))
    q = feat_query / feat_query.norm(dim=-1, keepdim=True)
    v = self._vox_feat / self._vox_feat.norm(dim=-1, keepdim=True)
    return dict(vox_xyz=self.global_vox_xyz, vox_sim=q @ v.T)


def _probe_server(labels=("person", "tree"), points=MISSION_POINTS,
                  prompts=MISSION_PROMPTS, dim=4, background=("tree",)):
  """A server wired for the probe log path only."""
  s = _bare_server(list(labels), feats=_FakeEncoder(dim)._feats(list(labels)))
  s.cfg = types.SimpleNamespace(
    querying=types.SimpleNamespace(compressed=False, text_query_mode="prompts"))
  s.encoder = _FakeEncoder(dim)
  s.feat_compressor = None
  s._probe_background = frozenset(background or ())
  s._probe_points_flu = tuple(mrc.parse_probe_points(points))
  if prompts:
    s._encode_probe_prompts(mrc.parse_probe_prompts(prompts))
  return s


def _voxels_at(flu_points):
  """Nx3 RDF voxel positions for a list of world-FLU points."""
  return torch.tensor([list(mrc.world_flu_to_rdf(p)) for p in flu_points],
                      dtype=torch.float32)


# -- registration ----------------------------------------------------------- #

def test_probe_prompts_are_encoded_once_and_held_out_of_the_query_set():
  s = _probe_server()
  assert list(s._probe_labels) == [
    "person lying down", "person lying on the ground", "casualty",
    "human body", "mannequin"]
  assert s._probe_feats.shape == (5, 4)
  # The mission's text_query_mode decides the path, same as add_queries.
  assert s.encoder.prompt_calls == [list(s._probe_labels)]
  assert s.encoder.label_calls == []
  # ...and NOTHING landed in the published query set.
  assert s._queries_labels["text"] == ["person", "tree"]
  assert s._queries_feats["text"].shape == (2, 4)
  assert s._queries_labels_history == {"person", "tree"}


def test_probe_prompts_follow_text_query_mode_labels():
  s = _probe_server(prompts=None)
  s.cfg.querying.text_query_mode = "labels"
  s._encode_probe_prompts(["casualty"])
  assert s.encoder.label_calls == [["casualty"]]
  assert s.encoder.prompt_calls == []


def test_a_bad_text_query_mode_disables_probes_instead_of_raising():
  s = _probe_server(prompts=None)
  s.cfg.querying.text_query_mode = "nonsense"
  s._encode_probe_prompts(["casualty"])
  assert s._probe_feats is None
  assert not s._probe_labels


def test_probes_are_disabled_without_a_text_capable_encoder():
  s = _probe_server(prompts=None)
  s.encoder = None
  s._encode_probe_prompts(["casualty"])
  assert s._probe_feats is None


def test_probes_never_fit_an_unfitted_feature_compressor():
  """Fitting a compressor on 5 diagnostic vectors would corrupt the map.

  CONTRACT CHANGE 2026-09-02 (with the store-queries-raw fix): probes are no
  longer DISABLED when the compressor is unfitted — like the vocabulary,
  they are stored RAW at encode time and compressed lazily by
  _log_raw_query_scores once the first frame has fitted the basis. The one
  invariant that must never move: encoding probes fits NOTHING.
  """
  class _Compressor:
    def __init__(self):
      self.fits = 0
      self.compressions = 0
    def is_fitted(self):
      return False
    def fit(self, x):
      self.fits += 1
    def compress(self, x):
      self.compressions += 1
      return x[:, :2]

  s = _probe_server(prompts=None)
  s.cfg.querying.compressed = True
  s.feat_compressor = _Compressor()
  s._encode_probe_prompts(["casualty"])
  assert s.feat_compressor.fits == 0
  assert s.feat_compressor.compressions == 0   # stored raw, not compressed
  assert s._probe_feats is not None
  assert s._probe_labels == ("casualty",)


def test_init_probes_prefers_the_env_var_over_the_config(monkeypatch):
  s = MultiRobotMappingServer.__new__(MultiRobotMappingServer)
  s.cfg = types.SimpleNamespace(
    querying=types.SimpleNamespace(compressed=False, text_query_mode="prompts"))
  s.encoder = _FakeEncoder()
  s.feat_compressor = None
  cfg = {"probe_prompts": "from_config", "probe_points": "[[9,9]]"}
  monkeypatch.setenv("RAYFRONTS_PROBE_PROMPTS", "from_env_a,from_env_b")
  monkeypatch.setenv("RAYFRONTS_PROBE_POINTS", "[[1,2]]")
  s._init_probes(types.SimpleNamespace(**cfg))
  assert list(s._probe_labels) == ["from_env_a", "from_env_b"]
  assert s._probe_points_flu == ((1.0, 2.0),)


def test_init_probes_falls_back_to_the_config_when_the_env_is_blank(
    monkeypatch):
  s = MultiRobotMappingServer.__new__(MultiRobotMappingServer)
  s.cfg = types.SimpleNamespace(
    querying=types.SimpleNamespace(compressed=False, text_query_mode="prompts"))
  s.encoder = _FakeEncoder()
  s.feat_compressor = None
  monkeypatch.setenv("RAYFRONTS_PROBE_PROMPTS", "   ")
  monkeypatch.delenv("RAYFRONTS_PROBE_POINTS", raising=False)
  s._init_probes(types.SimpleNamespace(probe_prompts=["casualty"],
                                       probe_points=[[3.0, 4.0, 5.0]]))
  assert list(s._probe_labels) == ["casualty"]
  assert s._probe_points_flu == ((3.0, 4.0, 5.0),)


def test_init_probes_with_nothing_set_leaves_the_probe_path_off(monkeypatch):
  s = MultiRobotMappingServer.__new__(MultiRobotMappingServer)
  s.cfg = types.SimpleNamespace(
    querying=types.SimpleNamespace(compressed=False, text_query_mode="prompts"))
  s.encoder = _FakeEncoder()
  s.feat_compressor = None
  monkeypatch.delenv("RAYFRONTS_PROBE_PROMPTS", raising=False)
  monkeypatch.delenv("RAYFRONTS_PROBE_POINTS", raising=False)
  s._init_probes(types.SimpleNamespace())
  assert s._probe_feats is None
  assert s._probe_points_flu == ()


# -- the nearest-voxel search ------------------------------------------------ #

def test_torch_nearest_vox_agrees_with_the_numpy_reference():
  s = _probe_server(prompts=None)
  torch.manual_seed(0)
  vox = (torch.rand(500, 3) - 0.5) * 200.0
  pts = mrc.parse_probe_points(MISSION_POINTS) + [(3.0, -4.0, 5.0)]
  for p in pts:
    k_t, d_t = s._nearest_vox(vox, p)
    k_n, d_n = mrc.nearest_index(vox.numpy(), mrc.world_flu_to_rdf(p),
                                 axes=mrc.probe_axes(p))
    assert k_t == k_n
    assert d_t == pytest.approx(d_n, abs=1e-3)


# -- the log itself ---------------------------------------------------------- #

def _run_log(s, caplog):
  caplog.set_level(logging.INFO,
                   logger="rayfronts.multi_robot_mapping_server")
  s._log_raw_query_scores()
  return [r.getMessage() for r in caplog.records]


def test_the_probe_path_never_mutates_the_published_query_set(caplog):
  """THE invariant: a probe must not move the softmax raven thresholds on."""
  s = _probe_server()
  s.mapper = _ProbeMapper(_voxels_at([[-12.9, -33.5, 1.0], [40.0, 40.0, 1.0]]),
                          torch.eye(2, 4))
  labels_before = list(s._queries_labels["text"])
  feats_before = s._queries_feats["text"].clone()
  history_before = set(s._queries_labels_history)

  _run_log(s, caplog)

  assert s._queries_labels["text"] == labels_before
  torch.testing.assert_close(s._queries_feats["text"], feats_before)
  assert s._queries_labels_history == history_before
  assert s._queries_updated is False
  # The probe rows rode along on ONE feature_query: 2 vocabulary + 5 probes.
  assert s.mapper.query_rows == [7]


def test_probe_point_rows_name_the_point_the_voxel_and_both_score_sets(caplog):
  s = _probe_server()
  s.mapper = _ProbeMapper(_voxels_at([[-12.9, -33.5, 1.0], [40.0, 40.0, 1.0]]),
                          torch.eye(2, 4))
  msgs = _run_log(s, caplog)

  hit = [m for m in msgs if m.startswith("[probe@(-12.87,-33.52)] vox#")]
  assert len(hit) == 1
  assert " labels: person=" in hit[0] and "tree=" in hit[0]
  assert "d=0.0" in hit[0]
  # The live positive carries raw/solo; the background label is raw only.
  assert re.search(r"person=\d\.\d{3}/\d\.\d{3}", hit[0])
  assert re.search(r"tree=\d\.\d{3}(?!/)", hit[0])

  probes = [m for m in msgs if m.startswith("[probe@(-12.87,-33.52)] probes:")]
  assert len(probes) == 1
  # Spaces become underscores so `k=v` stays parseable.
  for name in ("person_lying_down", "person_lying_on_the_ground", "casualty",
               "human_body", "mannequin"):
    assert f"{name}=" in probes[0]
  assert "person lying down=" not in probes[0]


def test_a_probe_point_with_no_voxel_nearby_says_so(caplog):
  s = _probe_server()
  s.mapper = _ProbeMapper(_voxels_at([[500.0, 500.0, 1.0]]), torch.eye(1, 4))
  msgs = _run_log(s, caplog)
  misses = [m for m in msgs if "no voxel within 3.0 m" in m]
  assert len(misses) == 3          # one per mission probe point
  assert "nearest " in misses[0] and " of 1 voxels" in misses[0]


def test_probe_top_row_extends_the_raw_top_block(caplog):
  s = _probe_server()
  s.mapper = _ProbeMapper(_voxels_at([[-12.9, -33.5, 1.0], [40.0, 40.0, 1.0]]),
                          torch.eye(2, 4))
  msgs = _run_log(s, caplog)
  tops = [m for m in msgs if m.startswith("[probe top-person] @(")]
  assert len(tops) == 1
  assert "casualty=" in tops[0] and "mannequin=" in tops[0]
  # The live positive leads as the reference column, every entry raw/solo...
  body = tops[0].split("): ", 1)[1]
  assert body.split()[0].startswith("person=")
  for tok in body.split():
    assert re.fullmatch(r"[^=\s]+=\d\.\d{3}/\d\.\d{3}", tok), tok
  # ...and the background label is NOT repeated here (it is the denominator).
  assert "tree=" not in tops[0]
  # A background label's own argmax gets no probe row.
  assert not [m for m in msgs if m.startswith("[probe top-tree]")]
  # The frozen [raw top] line is still emitted, unchanged in shape.
  assert any(m.startswith("[raw top] person") for m in msgs)
  assert any(m.startswith("[raw top] tree") for m in msgs)


def test_probe_top_is_logged_once_per_voxel_not_once_per_label(caplog):
  """7 mission labels mostly share one argmax; the row must not repeat."""
  labels = ["person", "casualty target", "road", "grass"]
  s = _probe_server(labels=tuple(labels), prompts="casualty", points=None,
                    background=("road", "grass"))
  # Both positives peak on voxel 0; the backgrounds peak on voxel 1.
  sim = torch.tensor([[0.30, 0.10], [0.28, 0.11], [0.10, 0.30], [0.09, 0.31],
                      [0.33, 0.12]], dtype=torch.float32)
  s.mapper = _FixedSimMapper(_voxels_at([[0.0, 0.0, 1.0], [9.0, 9.0, 1.0]]),
                             sim)
  msgs = _run_log(s, caplog)
  tops = [m for m in msgs if m.startswith("[probe top-")]
  assert len(tops) == 1
  assert tops[0].startswith("[probe top-person] ")


def test_probe_scores_are_the_real_cosines_of_the_probe_feats(caplog):
  """The numbers logged must be the probe feats' cosines, not the labels'."""
  s = _probe_server(labels=("person",))
  vox_feat = torch.tensor([[0.0, 1.0, 0.0, 0.0]])   # matches probe row 1
  s.mapper = _ProbeMapper(_voxels_at([[-12.87, -33.52, 1.0]]), vox_feat)
  msgs = _run_log(s, caplog)
  row = [m for m in msgs if m.startswith("[probe@(-12.87,-33.52)] probes:")][0]
  # probe feats: row i has 1.0 at i%4 and 0.25 at (i+1)%4 -> cos with e1 is
  # 0.25/sqrt(1.0625) for row 0 and 1/sqrt(1.0625) for row 1.
  norm = float(torch.tensor(1.0625).sqrt())
  assert f"person_lying_down={0.25 / norm:.3f}" in row
  assert f"person_lying_on_the_ground={1.0 / norm:.3f}" in row


def test_no_probes_configured_leaves_the_raw_log_exactly_as_it_was(caplog):
  s = _probe_server(points=None, prompts=None)
  s.mapper = _ProbeMapper(_voxels_at([[1.0, 1.0, 1.0]]), torch.eye(1, 4))
  msgs = _run_log(s, caplog)
  assert s.mapper.query_rows == [2]        # vocabulary only, nothing stacked
  assert not [m for m in msgs if m.startswith("[probe")]
  assert any(m.startswith("[raw vox] person") for m in msgs)


def test_a_probe_failure_cannot_take_down_the_raw_log(caplog):
  s = _probe_server()
  s.mapper = _ProbeMapper(_voxels_at([[-12.9, -33.5, 1.0]]), torch.eye(1, 4))
  s._probe_points_flu = ("not a point",)   # cannot happen via the parser
  msgs = _run_log(s, caplog)
  assert any(m.startswith("[raw vox] person") for m in msgs)
  assert any("probe-point logging failed" in m for m in msgs)


# -- the SOLO softmax ------------------------------------------------------- #

def _solo_reference(cos_i, bg_cosines, temp=100.0):
  """The formula, written out: exp(T*c_i) / (exp(T*c_i) + sum_b exp(T*c_b)).

  Computed in plain python floats with a max-shift for stability, so it shares
  no code with the torch implementation under test.
  """
  zs = [temp * cos_i] + [temp * c for c in bg_cosines]
  m = max(zs)
  es = [math.exp(z - m) for z in zs]
  return es[0] / sum(es)


def test_solo_softmax_matches_a_hand_written_reference():
  s = _probe_server(prompts=None, points=None)
  # rows: 0 person (positive), 1 road (bg), 2 grass (bg), 3 probe casualty.
  sim = torch.tensor([[0.19], [0.21], [0.15], [0.24]], dtype=torch.float32)
  solo = s._solo_probs(sim, 0, solo_rows=[0, 3], bg_rows=[1, 2])
  assert set(solo) == {0, 3}
  assert solo[0] == pytest.approx(_solo_reference(0.19, [0.21, 0.15]), rel=1e-5)
  assert solo[3] == pytest.approx(_solo_reference(0.24, [0.21, 0.15]), rel=1e-5)
  # And it is a probability against that contrast set, not a cosine.
  assert 0.0 < solo[0] < 1.0
  assert solo[3] > solo[0]        # the better-matching wording scores higher


def test_solo_softmax_uses_the_live_temperature():
  """100 is rayfronts/utils.py compute_cos_sim's `torch.softmax(100 * sim)`."""
  assert MultiRobotMappingServer.PROBE_SOFTMAX_TEMP == 100.0
  s = _probe_server(prompts=None, points=None)
  sim = torch.tensor([[0.30], [0.20]], dtype=torch.float32)
  solo = s._solo_probs(sim, 0, solo_rows=[0], bg_rows=[1])
  assert solo[0] == pytest.approx(_solo_reference(0.30, [0.20], temp=100.0),
                                  rel=1e-5)
  # A joint softmax at temperature 1 would give ~0.52; the live one gives
  # ~0.73. Reading the wrong one off the log is a 0.6-gate-sized error.
  assert solo[0] != pytest.approx(_solo_reference(0.30, [0.20], temp=1.0),
                                  rel=1e-3)


def test_a_sibling_probe_does_not_change_another_probes_solo_score():
  """Near-synonyms must not split each other's mass — the whole point."""
  s = _probe_server(prompts=None, points=None)
  one = torch.tensor([[0.19], [0.21], [0.24]], dtype=torch.float32)
  two = torch.tensor([[0.19], [0.21], [0.24], [0.245]], dtype=torch.float32)
  a = s._solo_probs(one, 0, solo_rows=[0, 2], bg_rows=[1])
  b = s._solo_probs(two, 0, solo_rows=[0, 2, 3], bg_rows=[1])
  assert b[2] == pytest.approx(a[2], rel=1e-9)   # the probe is unmoved
  assert b[0] == pytest.approx(a[0], rel=1e-9)   # so is the live positive


def test_the_denominator_is_exactly_the_background_rows():
  """A live positive present in `labels` must not leak into the contrast set."""
  s = _probe_server(prompts=None, points=None)
  # person (positive) scores very high; if it leaked into the denominator the
  # probe's solo score would collapse.
  sim = torch.tensor([[0.95], [0.10], [0.12], [0.30]], dtype=torch.float32)
  solo = s._solo_probs(sim, 0, solo_rows=[0, 3], bg_rows=[1, 2])
  assert solo[3] == pytest.approx(_solo_reference(0.30, [0.10, 0.12]), rel=1e-5)
  leaked = _solo_reference(0.30, [0.10, 0.12, 0.95])
  assert solo[3] != pytest.approx(leaked, rel=1e-3)
  assert solo[3] > 0.9 and leaked < 0.01


def test_row_sets_split_the_vocabulary_by_the_contrast_set():
  s = _probe_server(labels=("person", "road", "grass"), prompts=None,
                    points=None, background=("road", "grass"))
  bg, solo = s._probe_row_sets(["person", "road", "grass"], 2)
  assert bg == [1, 2]                 # exactly the background labels
  assert solo == [0, 3, 4]            # the positive, then the two probes


def test_no_contrast_set_means_raw_only_never_a_fake_solo_of_one(caplog):
  """With an empty background the softmax would be exactly 1.0 for every row."""
  s = _probe_server(background=())
  s.mapper = _ProbeMapper(_voxels_at([[-12.87, -33.52, 1.0]]), torch.eye(1, 4))
  msgs = _run_log(s, caplog)
  row = [m for m in msgs if m.startswith("[probe@(-12.87,-33.52)] probes:")][0]
  assert "/" not in row
  assert "1.000" not in row


def test_init_probes_resolves_the_contrast_set_from_env_then_query_file(
    monkeypatch):
  s = MultiRobotMappingServer.__new__(MultiRobotMappingServer)
  s.cfg = types.SimpleNamespace(
    querying=types.SimpleNamespace(compressed=False, text_query_mode="prompts"))
  s.encoder = _FakeEncoder()
  s.feat_compressor = None
  monkeypatch.setenv("RAYFRONTS_PROBE_PROMPTS", "casualty")
  monkeypatch.delenv("RAYFRONTS_PROBE_POINTS", raising=False)

  # 1. the env var wins.
  monkeypatch.setenv("RAYFRONTS_PROBE_BACKGROUND", "road,grass,tree")
  s._init_probes(types.SimpleNamespace(), seeded_labels=["sky"])
  assert s._probe_background == frozenset({"road", "grass", "tree"})

  # 2. unset -> whatever querying.query_file seeded at startup.
  monkeypatch.delenv("RAYFRONTS_PROBE_BACKGROUND", raising=False)
  s._init_probes(types.SimpleNamespace(), seeded_labels=["sky", "road"])
  assert s._probe_background == frozenset({"sky", "road"})

  # 3. neither -> empty, and the raw-only warning path.
  s._init_probes(types.SimpleNamespace(), seeded_labels=[])
  assert s._probe_background == frozenset()


class _FixedSimMapper:
  """A mapper that returns a caller-supplied [Q, N] sim matrix verbatim.

  Real RADSeg cosines live in a narrow band (~0.15-0.25 measured live), which
  is exactly the band where the temperature-100 softmax does its work; hand
  cosines in directly rather than trying to build unit vectors that produce
  them.
  """

  def __init__(self, vox_xyz, sim):
    self.global_vox_xyz = vox_xyz
    self.global_rays_orig_angles = None
    self._sim = sim
    self.query_rows = []

  def feature_query(self, feat_query, softmax=False, compressed=True):
    self.query_rows.append(int(feat_query.shape[0]))
    return dict(vox_xyz=self.global_vox_xyz, vox_sim=self._sim)


def test_solo_reaches_the_gate_reading_the_operator_needs(caplog):
  """A row must read as 'person below 0.6, casualty above it'."""
  s = _probe_server(labels=("person", "road", "grass"), prompts="casualty",
                    points='[[0,0]]', background=("road", "grass"))
  # rows: person 0.19, road 0.21, grass 0.15, casualty 0.24 — the live band.
  sim = torch.tensor([[0.19], [0.21], [0.15], [0.24]], dtype=torch.float32)
  s.mapper = _FixedSimMapper(_voxels_at([[0.0, 0.0, 1.0]]), sim)
  msgs = _run_log(s, caplog)

  labels_row = [m for m in msgs if " labels: person=" in m][0]
  probes_row = [m for m in msgs if m.startswith("[probe@(0.00,0.00)] probes:")]
  person_solo = float(re.search(r"person=[\d.]+/([\d.]+)",
                                labels_row).group(1))
  casualty_solo = float(re.search(r"casualty=[\d.]+/([\d.]+)",
                                  probes_row[0]).group(1))
  assert person_solo == pytest.approx(_solo_reference(0.19, [0.21, 0.15]),
                                      abs=5e-4)
  assert casualty_solo == pytest.approx(_solo_reference(0.24, [0.21, 0.15]),
                                        abs=5e-4)
  # The reading the operator needs: this wording crosses the live 0.6 gate
  # and `person` does not, on the SAME voxel.
  assert person_solo < 0.6 < casualty_solo


def test_a_pruned_map_between_query_and_log_is_skipped_not_misindexed(caplog):
  """vox_sim and global_vox_xyz disagreeing must not log a wrong voxel."""
  s = _probe_server()
  s.mapper = _ProbeMapper(_voxels_at([[-12.9, -33.5, 1.0], [40.0, 40.0, 1.0]]),
                          torch.eye(2, 4))
  real = s.mapper.feature_query

  def _short(feat_query, softmax=False, compressed=True):
    r = real(feat_query, softmax=softmax, compressed=compressed)
    r["vox_xyz"] = r["vox_xyz"][:1]      # a prune landed after the query
    return r

  s.mapper.feature_query = _short
  msgs = _run_log(s, caplog)
  assert not [m for m in msgs if m.startswith("[probe@")]
