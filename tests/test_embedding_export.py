"""The rules-planner embedding export: basis, clouds, meta, text service.

``rayfronts.embedding_export`` imports nothing from ``rayfronts``, so it is
loaded straight off disk (``conftest.load_standalone``) and everything here
runs with torch alone -- no ROS, no encoder, no compiled ``rayfronts_cpp``.

The messaging service is faked with the SAME numpy recarray construction
``messaging_services/ros.py`` uses, so the field names, their order and their
dtype are asserted against the layout that actually goes on the wire.
"""

import json
import pathlib
import types

import numpy as np
import pytest

from conftest import HAVE_TORCH, PKG, load_standalone

pytestmark = [pytest.mark.torch,
              pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")]

if HAVE_TORCH:
  import torch
  ee = load_standalone("embedding_export")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeMessaging:
  """Records every topic a Ros2MessagingService would have created.

  ``has_subscribers`` registers the layer the way the real one does (it goes
  through ``_get_publisher``, which creates the publisher lazily), so an
  empty ``publishers`` set really does mean "no new topic".
  """

  def __init__(self, subscribed=True):
    self.subscribed = subscribed
    self.publishers = set()
    self.subscriptions = dict()
    self.clouds = list()
    self.strings = list()

  def has_subscribers(self, layer):
    self.publishers.add(layer)
    return self.subscribed

  def publish_pc(self, pc_xyz, features=None, layer="pc"):
    self.publishers.add(layer)
    n = int(pc_xyz.shape[0])
    dtype_list = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
    for name in (features or dict()):
      dtype_list.append((name, np.float32))
    rec = np.recarray((n,), dtype=dtype_list)
    xyz = pc_xyz.cpu().numpy().astype(np.float32)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, tensor in (features or dict()).items():
      rec[name] = tensor.cpu().numpy().astype(np.float32)
    self.clouds.append((layer, rec))

  def publish_string(self, layer, data, latched=False):
    self.publishers.add(layer)
    self.strings.append((layer, data, latched))

  def subscribe_string(self, layer, callback):
    self.subscriptions[layer] = callback

  def cloud(self, layer):
    for name, rec in self.clouds:
      if name == layer:
        return rec
    return None

  def last_string(self, layer):
    for name, data, latched in reversed(self.strings):
      if name == layer:
        return data
    return None


class FakeEncoder:
  """A fixed linear 'language head' and deterministic text vectors."""

  model_version = "fake-radio"
  lang_model = "fake-siglip"

  def __init__(self, in_dim=6, lang_dim=8):
    self.in_dim = in_dim
    self.lang_dim = lang_dim
    g = torch.Generator().manual_seed(7)
    self._head = torch.randn(in_dim, lang_dim, generator=g)
    self.prompt_calls = list()
    self.label_calls = list()

  def align_spatial_features_with_language(self, features):
    b, c, h, w = features.shape
    x = features.permute(0, 2, 3, 1).reshape(b, -1, c)
    out = x @ self._head
    return out.permute(0, 2, 1).reshape(b, -1, h, w)

  def _text(self, texts):
    out = torch.zeros(len(texts), self.lang_dim)
    for i, t in enumerate(texts):
      g = torch.Generator().manual_seed(abs(hash(str(t))) % (2 ** 31))
      out[i] = torch.randn(self.lang_dim, generator=g)
    return torch.nn.functional.normalize(out, dim=-1)

  def encode_prompts(self, prompts):
    self.prompt_calls.append(list(prompts))
    return self._text(prompts)

  def encode_labels(self, labels):
    self.label_calls.append(list(labels))
    return self._text(labels)


class FakeCompressor:
  def __init__(self, in_dim, out_dim):
    g = torch.Generator().manual_seed(3)
    self.basis = torch.randn(in_dim, out_dim, generator=g)
    self.calls = 0

  def is_fitted(self):
    return True

  def decompress(self, y):
    self.calls += 1
    return y @ self.basis.T


class FakeMapper:
  """Same tensor layout SemanticRayFrontiersMap exposes to feature_query."""

  vox_size = 0.3

  def __init__(self, n_vox=40, n_ray=5, dim=6, feat_compressor=None):
    g = torch.Generator().manual_seed(11)
    self.feat_compressor = feat_compressor
    self.global_vox_xyz = (torch.randn(n_vox, 3, generator=g)
                           if n_vox else None)
    self.global_vox_feat = (torch.randn(n_vox, dim, generator=g)
                            if n_vox else None)
    self.global_vox_cnt = (torch.arange(1, n_vox + 1).reshape(-1, 1).float()
                           if n_vox else None)
    self.global_rays_orig_angles = (torch.randn(n_ray, 5, generator=g)
                                    if n_ray else None)
    self.global_rays_feat = (torch.randn(n_ray, dim, generator=g)
                             if n_ray else None)
    self.global_rays_cnt = (torch.arange(1, n_ray + 1).reshape(-1, 1).float()
                            if n_ray else None)


def make_cfg(enabled=True, k=4, period=None, seed_vocab=None,
             fit_min_voxels=5, save_dir=None, text_query_mode="prompts",
             querying_period=10):
  return dict(
    querying=dict(period=querying_period, text_query_mode=text_query_mode),
    emb=dict(enabled=enabled, k=k, period=period, seed_vocab=seed_vocab,
             fit_min_voxels=fit_min_voxels, save_dir=save_dir))


@pytest.fixture
def exporter(tmp_path):
  """A started, enabled exporter over a small fake map."""
  ms = FakeMessaging()
  enc = FakeEncoder()
  mapper = FakeMapper()
  exp = ee.EmbeddingExporter(
    make_cfg(save_dir=str(tmp_path)), mapper=mapper, encoder=enc,
    messaging_service=ms)
  exp.start()
  exp.messaging = ms
  exp.enc = enc
  exp.fake_mapper = mapper
  return exp


def _aligned(mapper, encoder):
  return ee.aligned_embeddings(mapper, encoder)


# --------------------------------------------------------------------------- #
# EmbeddingProjector
# --------------------------------------------------------------------------- #

def test_fit_produces_a_k_by_d_basis_and_project_is_l2_normalised():
  x = torch.nn.functional.normalize(torch.randn(200, 16), dim=-1)
  p = ee.EmbeddingProjector(k=5)
  p.fit(x)
  assert p.is_fitted()
  assert p.mean.shape == (16,)
  assert p.basis.shape == (5, 16)
  assert p.dim == 16
  assert p.fit_n == 200
  y = p.project(x)
  assert y.shape == (200, 5)
  torch.testing.assert_close(y.norm(dim=-1), torch.ones(200), atol=1e-5,
                             rtol=1e-5)


def test_seed_text_joins_the_fit_and_moves_the_basis():
  x = torch.nn.functional.normalize(torch.randn(100, 16), dim=-1)
  seed = torch.nn.functional.normalize(torch.randn(30, 16), dim=-1)
  a, b = ee.EmbeddingProjector(k=4), ee.EmbeddingProjector(k=4)
  a.fit(x)
  b.fit(x, seed_text=seed)
  assert a.fit_n == 100 and b.fit_n == 130
  assert not torch.allclose(a.mean, b.mean)


def test_k_is_clamped_to_what_the_data_allows():
  x = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
  p = ee.EmbeddingProjector(k=128)
  p.fit(x)
  assert p.k == 6
  assert p.basis.shape == (6, 16)
  assert p.project(x).shape == (6, 6)


def test_project_before_fit_is_an_error_not_a_silent_zero():
  with pytest.raises(RuntimeError):
    ee.EmbeddingProjector(k=4).project(torch.randn(3, 8))


def test_cos_preservation_is_near_one_when_k_equals_the_rank():
  g = torch.Generator().manual_seed(5)
  x = torch.nn.functional.normalize(torch.randn(120, 8, generator=g), dim=-1)
  p = ee.EmbeddingProjector(k=8)
  p.fit(x)
  r = p.cos_preservation(x)
  assert 0.95 <= r <= 1.0 + 1e-6


def test_cos_preservation_drops_when_the_basis_is_too_small():
  g = torch.Generator().manual_seed(5)
  x = torch.nn.functional.normalize(torch.randn(120, 32, generator=g), dim=-1)
  wide = ee.EmbeddingProjector(k=32)
  wide.fit(x)
  narrow = ee.EmbeddingProjector(k=2)
  narrow.fit(x)
  assert narrow.cos_preservation(x) < wide.cos_preservation(x)


def test_cos_preservation_subsamples_large_inputs():
  x = torch.nn.functional.normalize(torch.randn(2500, 8), dim=-1)
  p = ee.EmbeddingProjector(k=6)
  p.fit(x)
  r = p.cos_preservation(x, sample=50)
  assert -1.0 - 1e-6 <= r <= 1.0 + 1e-6


def test_cos_preservation_of_a_degenerate_input_is_nan_not_a_crash():
  x = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
  p = ee.EmbeddingProjector(k=2)
  p.fit(x)
  assert np.isnan(p.cos_preservation(x))


def test_save_load_round_trip_reproduces_the_projection(tmp_path):
  x = torch.nn.functional.normalize(torch.randn(80, 16), dim=-1)
  p = ee.EmbeddingProjector(k=5)
  p.fit(x)
  p.fit_cos_preservation = p.cos_preservation(x)
  fp = str(tmp_path / "emb_basis.pt")
  p.save(fp)

  q = ee.EmbeddingProjector(k=1)
  q.load(fp)
  assert q.k == 5 and q.fit_n == 80
  assert q.fit_cos_preservation == pytest.approx(p.fit_cos_preservation)
  torch.testing.assert_close(q.project(x), p.project(x))


def test_save_creates_the_directory(tmp_path):
  p = ee.EmbeddingProjector(k=2)
  p.fit(torch.nn.functional.normalize(torch.randn(10, 4), dim=-1))
  fp = str(tmp_path / "nested" / "dir" / "emb_basis.pt")
  p.save(fp)
  assert pathlib.Path(fp).exists()


# --------------------------------------------------------------------------- #
# aligned_embeddings
# --------------------------------------------------------------------------- #

def test_aligned_embeddings_shapes_and_normalisation():
  mapper = FakeMapper(n_vox=12, n_ray=4, dim=6)
  enc = FakeEncoder(in_dim=6, lang_dim=8)
  vox, rays = _aligned(mapper, enc)

  vox_xyz, vox_cnt, vox_emb = vox
  assert vox_xyz.shape == (12, 3)
  assert vox_cnt.shape == (12,)
  assert vox_emb.shape == (12, 8)
  torch.testing.assert_close(vox_emb.norm(dim=-1), torch.ones(12), atol=1e-5,
                             rtol=1e-5)

  roa, ray_cnt, ray_emb = rays
  assert roa.shape == (4, 5)
  assert ray_cnt.shape == (4,)
  assert ray_emb.shape == (4, 8)
  torch.testing.assert_close(ray_emb.norm(dim=-1), torch.ones(4), atol=1e-5,
                             rtol=1e-5)


def test_aligned_embeddings_matches_the_feature_query_path():
  """Same decompress -> unsqueeze -> align -> normalise feature_query runs."""
  mapper = FakeMapper(n_vox=7, n_ray=0, dim=6)
  enc = FakeEncoder(in_dim=6, lang_dim=8)
  expected = torch.nn.functional.normalize(
    enc.align_spatial_features_with_language(
      mapper.global_vox_feat.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1), dim=-1)
  vox, _ = _aligned(mapper, enc)
  torch.testing.assert_close(vox[2], expected)


def test_aligned_embeddings_decompresses_a_fitted_compressor():
  fc = FakeCompressor(in_dim=6, out_dim=3)
  mapper = FakeMapper(n_vox=9, n_ray=2, dim=3, feat_compressor=fc)
  enc = FakeEncoder(in_dim=6, lang_dim=8)
  vox, rays = _aligned(mapper, enc)
  assert fc.calls == 2                      # voxels and rays
  assert vox[2].shape == (9, 8)
  assert rays[2].shape == (2, 8)


def test_aligned_embeddings_on_an_empty_map_and_a_map_without_rays():
  enc = FakeEncoder()
  assert _aligned(FakeMapper(n_vox=0, n_ray=0), enc) == (None, None)
  vox, rays = _aligned(FakeMapper(n_vox=3, n_ray=0), enc)
  assert vox is not None and rays is None
  assert _aligned(None, enc) == (None, None)
  assert _aligned(FakeMapper(), None) == (None, None)


def test_count_defaults_to_ones_when_the_mapper_has_none():
  mapper = FakeMapper(n_vox=4, n_ray=0)
  mapper.global_vox_cnt = None
  vox, _ = _aligned(mapper, FakeEncoder())
  torch.testing.assert_close(vox[1], torch.ones(4))


# --------------------------------------------------------------------------- #
# Published clouds
# --------------------------------------------------------------------------- #

def test_voxel_cloud_field_layout(exporter):
  exporter.run_once()
  rec = exporter.messaging.cloud(ee.VOX_LAYER)
  assert rec is not None
  assert rec.dtype.names == ("x", "y", "z", "cnt",
                             "e_0", "e_1", "e_2", "e_3")
  for name in rec.dtype.names:
    assert rec.dtype[name] == np.float32
  assert rec.shape == (40,)
  np.testing.assert_allclose(rec["cnt"], np.arange(1, 41), rtol=1e-6)
  # Every published vector is a unit vector: dot(text, voxel) is the cosine.
  emb = np.stack([rec[f"e_{j}"] for j in range(4)], axis=1)
  np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(40),
                             atol=1e-5)


def test_ray_cloud_field_layout_and_angles(exporter):
  exporter.run_once()
  rec = exporter.messaging.cloud(ee.RAY_LAYER)
  assert rec.dtype.names == ("x", "y", "z", "theta", "phi", "cnt",
                             "e_0", "e_1", "e_2", "e_3")
  for name in rec.dtype.names:
    assert rec.dtype[name] == np.float32
  assert rec.shape == (5,)
  roa = exporter.fake_mapper.global_rays_orig_angles.numpy()
  # Origin + angles, the same convention rays_sim uses.
  np.testing.assert_allclose(rec["x"], roa[:, 0], rtol=1e-5)
  np.testing.assert_allclose(rec["theta"], roa[:, 3], rtol=1e-5)
  np.testing.assert_allclose(rec["phi"], roa[:, 4], rtol=1e-5)


def test_meta_is_latched_json_with_the_frozen_key_set(exporter):
  exporter.run_once()
  layer, data, latched = [s for s in exporter.messaging.strings
                          if s[0] == ee.META_LAYER][0]
  assert layer == "emb/meta"
  assert latched is True
  meta = json.loads(data)
  assert tuple(meta.keys()) == ee.META_KEYS
  assert meta["k"] == 4
  assert meta["dim"] == 8
  assert meta["encoder"] == "fake-radio"
  assert meta["lang_model"] == "fake-siglip"
  assert meta["fit_n"] == 40
  assert 0.0 <= meta["cos_preservation"] <= 1.0
  assert meta["vox_size"] == pytest.approx(0.3)
  assert meta["query_mode"] == "prompts"


def test_meta_is_published_once_per_fit(exporter):
  exporter.run_once()
  exporter.run_once()
  exporter.run_once()
  metas = [s for s in exporter.messaging.strings if s[0] == ee.META_LAYER]
  assert len(metas) == 1
  # ...while the clouds keep coming.
  assert len([c for c in exporter.messaging.clouds
              if c[0] == ee.VOX_LAYER]) == 3


def test_the_basis_is_fitted_once_and_reused(exporter):
  exporter.run_once()
  basis = exporter.projector.basis.clone()
  exporter.fake_mapper.global_vox_feat = torch.randn(40, 6)
  exporter.run_once()
  torch.testing.assert_close(exporter.projector.basis, basis)


def test_nothing_is_published_before_fit_min_voxels(tmp_path):
  ms = FakeMessaging()
  exp = ee.EmbeddingExporter(
    make_cfg(fit_min_voxels=100, save_dir=str(tmp_path)),
    mapper=FakeMapper(n_vox=40), encoder=FakeEncoder(),
    messaging_service=ms)
  exp.start()
  exp.run_once()
  assert ms.clouds == []
  assert not exp.projector.is_fitted()


def test_publishing_is_subscriber_gated(tmp_path):
  ms = FakeMessaging(subscribed=False)
  exp = ee.EmbeddingExporter(
    make_cfg(save_dir=str(tmp_path)), mapper=FakeMapper(),
    encoder=FakeEncoder(), messaging_service=ms)
  exp.start()
  exp.run_once()
  assert ms.clouds == []
  assert not exp.projector.is_fitted()


def test_the_basis_file_is_written_to_save_dir(tmp_path, exporter):
  exporter.run_once()
  fp = pathlib.Path(exporter.save_dir) / ee.BASIS_FILENAME
  assert fp.exists()
  loaded = ee.EmbeddingProjector(k=1)
  loaded.load(str(fp))
  torch.testing.assert_close(loaded.basis, exporter.projector.basis)


def test_maybe_run_follows_emb_period(tmp_path):
  ms = FakeMessaging()
  exp = ee.EmbeddingExporter(
    make_cfg(period=3, querying_period=10, save_dir=str(tmp_path)),
    mapper=FakeMapper(), encoder=FakeEncoder(), messaging_service=ms)
  exp.start()
  assert exp.period == 3
  for i in range(7):
    exp.maybe_run(i)
  assert len([c for c in ms.clouds if c[0] == ee.VOX_LAYER]) == 3  # 0, 3, 6


def test_period_falls_back_to_querying_period(tmp_path):
  exp = ee.EmbeddingExporter(make_cfg(period=None, querying_period=17),
                             messaging_service=FakeMessaging())
  assert exp.period == 17


def test_period_zero_turns_publishing_off_without_falling_back():
  ms = FakeMessaging()
  exp = ee.EmbeddingExporter(make_cfg(period=0, querying_period=10),
                             mapper=FakeMapper(), encoder=FakeEncoder(),
                             messaging_service=ms)
  exp.start()
  for i in range(20):
    exp.maybe_run(i)
  assert ms.clouds == []


def test_maybe_run_never_raises_into_the_mapping_loop(tmp_path):
  class _Exploding(FakeMessaging):
    def publish_pc(self, *a, **kw):
      raise RuntimeError("boom")

  exp = ee.EmbeddingExporter(
    make_cfg(save_dir=str(tmp_path)), mapper=FakeMapper(),
    encoder=FakeEncoder(), messaging_service=_Exploding())
  exp.start()
  exp.maybe_run(0)          # must not propagate


def test_status_fields_track_the_fit(exporter):
  before = exporter.status_fields()
  assert before["fit_n"] == 0 and before["cos_preservation"] is None
  exporter.run_once()
  after = exporter.status_fields()
  assert after["k"] == 4 and after["fit_n"] == 40
  assert 0.0 <= after["cos_preservation"] <= 1.0


# --------------------------------------------------------------------------- #
# Seed vocabulary
# --------------------------------------------------------------------------- #

def test_seed_vocab_is_encoded_and_joins_the_fit(tmp_path):
  vocab = tmp_path / "vocab.txt"
  vocab.write_text("person\n# a comment\n\nroad\nperson\ndebris\n")
  ms = FakeMessaging()
  enc = FakeEncoder()
  exp = ee.EmbeddingExporter(
    make_cfg(seed_vocab=str(vocab), save_dir=str(tmp_path)),
    mapper=FakeMapper(n_vox=40), encoder=enc, messaging_service=ms)
  exp.start()
  exp.run_once()
  # Comments, blanks and duplicates dropped; order kept.
  assert enc.prompt_calls == [["person", "road", "debris"]]
  assert exp.projector.fit_n == 43


def test_seed_vocab_uses_encode_labels_in_labels_mode(tmp_path):
  vocab = tmp_path / "vocab.txt"
  vocab.write_text("person\nroad\n")
  enc = FakeEncoder()
  exp = ee.EmbeddingExporter(
    make_cfg(seed_vocab=str(vocab), save_dir=str(tmp_path),
             text_query_mode="labels"),
    mapper=FakeMapper(n_vox=40), encoder=enc,
    messaging_service=FakeMessaging())
  exp.start()
  exp.run_once()
  assert enc.label_calls == [["person", "road"]]
  assert enc.prompt_calls == []


def test_a_missing_seed_vocab_degrades_to_a_map_only_fit(tmp_path):
  exp = ee.EmbeddingExporter(
    make_cfg(seed_vocab=str(tmp_path / "nope.txt"), save_dir=str(tmp_path)),
    mapper=FakeMapper(n_vox=40), encoder=FakeEncoder(),
    messaging_service=FakeMessaging())
  exp.start()
  exp.run_once()
  assert exp.projector.is_fitted()
  assert exp.projector.fit_n == 40


# --------------------------------------------------------------------------- #
# Text request / response
# --------------------------------------------------------------------------- #

def _request(exp, **payload):
  exp.on_text_request(json.dumps(payload))
  return json.loads(exp.messaging.last_string(ee.TEXT_RESPONSE_LAYER))


def test_the_request_topic_is_subscribed_when_enabled(exporter):
  assert ee.TEXT_REQUEST_LAYER in exporter.messaging.subscriptions
  assert exporter.messaging.subscriptions[ee.TEXT_REQUEST_LAYER] == \
      exporter.on_text_request


def test_text_request_round_trip(exporter):
  exporter.run_once()
  r = _request(exporter, id="q1", phrases=["person", "road"], mode="prompts")
  assert r["id"] == "q1"
  assert r["k"] == 4
  assert r["phrases"] == ["person", "road"]
  v = torch.tensor(r["vectors"])
  assert v.shape == (2, 4)
  torch.testing.assert_close(v.norm(dim=-1), torch.ones(2), atol=1e-5,
                             rtol=1e-5)
  # Same basis as the clouds: the dot product is the cosine the planner uses.
  expected = exporter.projector.project(exporter.enc.encode_prompts(
    ["person", "road"]))
  torch.testing.assert_close(v, expected, atol=1e-5, rtol=1e-5)


def test_mode_defaults_to_prompts_and_labels_is_honoured(exporter):
  exporter.run_once()
  exporter.enc.prompt_calls.clear()
  _request(exporter, id="a", phrases=["person"])
  assert exporter.enc.prompt_calls == [["person"]]
  _request(exporter, id="b", phrases=["person"], mode="labels")
  assert exporter.enc.label_calls == [["person"]]


def test_responses_come_back_in_order(exporter):
  exporter.run_once()
  for i in range(5):
    exporter.on_text_request(json.dumps({"id": f"q{i}", "phrases": ["a"]}))
  ids = [json.loads(d)["id"] for l, d, _ in exporter.messaging.strings
         if l == ee.TEXT_RESPONSE_LAYER]
  assert ids == ["q0", "q1", "q2", "q3", "q4"]


def test_a_request_before_the_fit_gets_projector_not_fitted(exporter):
  r = _request(exporter, id="early", phrases=["person"])
  assert r == {"id": "early", "error": "projector not fitted"}


def test_malformed_json_is_answered_not_dropped(exporter):
  exporter.on_text_request("not json at all")
  r = json.loads(exporter.messaging.last_string(ee.TEXT_RESPONSE_LAYER))
  assert r["id"] is None and r["error"] == "malformed json"


def test_a_json_scalar_is_a_malformed_request(exporter):
  exporter.on_text_request("[1, 2, 3]")
  r = json.loads(exporter.messaging.last_string(ee.TEXT_RESPONSE_LAYER))
  assert r["error"] == "malformed request"


@pytest.mark.parametrize("phrases", [None, [], "person", 3])
def test_missing_or_empty_phrases_is_an_error(exporter, phrases):
  exporter.run_once()
  r = _request(exporter, id="x", phrases=phrases)
  assert r == {"id": "x", "error": "no phrases"}


def test_a_huge_request_is_refused(exporter):
  exporter.run_once()
  r = _request(exporter, id="big",
               phrases=["a"] * (ee.MAX_TEXT_PHRASES + 1))
  assert r["id"] == "big"
  assert "too many phrases" in r["error"]
  # The cap itself is still served.
  ok = _request(exporter, id="edge", phrases=["a"] * ee.MAX_TEXT_PHRASES)
  assert ok["k"] == 4


def test_an_unknown_mode_is_an_error(exporter):
  exporter.run_once()
  r = _request(exporter, id="m", phrases=["person"], mode="embeddings")
  assert r["id"] == "m" and "unknown mode" in r["error"]


def test_an_encoder_failure_is_reported_not_raised(exporter):
  exporter.run_once()

  def _boom(_):
    raise RuntimeError("no gpu")

  exporter.encoder.encode_prompts = _boom
  r = _request(exporter, id="e", phrases=["person"])
  assert r == {"id": "e", "error": "encoding failed"}


def test_bytes_payloads_are_accepted(exporter):
  exporter.run_once()
  exporter.on_text_request(json.dumps({"id": "b", "phrases": ["x"]}).encode())
  r = json.loads(exporter.messaging.last_string(ee.TEXT_RESPONSE_LAYER))
  assert r["id"] == "b" and r["k"] == 4


# --------------------------------------------------------------------------- #
# Disabled by default
# --------------------------------------------------------------------------- #

def test_disabled_creates_no_publisher_and_no_subscription():
  ms = FakeMessaging()
  exp = ee.EmbeddingExporter(make_cfg(enabled=False), mapper=FakeMapper(),
                             encoder=FakeEncoder(), messaging_service=ms)
  exp.start()
  for i in range(30):
    exp.maybe_run(i)
  exp.run_once()
  exp.on_text_request(json.dumps({"id": "x", "phrases": ["person"]}))

  assert ms.publishers == set()
  assert ms.subscriptions == dict()
  assert ms.clouds == []
  assert ms.strings == []
  assert exp.status_fields() is None
  assert not exp.projector.is_fitted()


def test_a_config_without_an_emb_section_is_disabled():
  for cfg in (dict(querying=dict(period=10)),
              types.SimpleNamespace(querying=dict(period=10)),
              dict(),
              None):
    exp = ee.EmbeddingExporter(cfg, messaging_service=FakeMessaging())
    assert exp.enabled is False
    assert exp.status_fields() is None


def test_default_yaml_ships_the_section_disabled():
  yaml = pytest.importorskip("yaml", reason="needs pyyaml")
  with open(PKG / "configs" / "default.yaml", "r", encoding="UTF-8") as f:
    cfg = yaml.safe_load(f)
  emb = cfg["emb"]
  assert emb["enabled"] is False
  assert set(emb) == {"enabled", "k", "period", "seed_vocab",
                      "fit_min_voxels", "save_dir"}
  assert emb["k"] == 128
  assert emb["period"] is None
  assert emb["seed_vocab"] is None
  assert emb["fit_min_voxels"] == 500
  assert emb["save_dir"] is None
