"""Language-aligned embedding export for the rules planner.

The planner never sends RayFronts a text query (see
``task_assigner/docs/rules_planner_embeddings.md``). Instead the mapper
publishes the map's language-aligned embeddings, projected into one shared
K-dim basis, and the planner clusters and labels them itself:

* ``emb/voxels`` -- ``x,y,z,cnt,e_0..e_{K-1}`` per semantic voxel;
* ``emb/rays``   -- ``x,y,z,theta,phi,cnt,e_0..e_{K-1}`` per semantic ray;
* ``emb/meta``   -- the basis metadata, latched;
* ``emb/text/request`` / ``emb/text/response`` -- phrases in, projected
  vectors out, so ``dot(text, voxel)`` is the cosine the planner thresholds.

Everything that is not ROS lives here so :class:`rayfronts.mapping_server.
MappingServer` and :class:`rayfronts.multi_robot_mapping_server.
MultiRobotMappingServer` share one implementation.

This module imports nothing from ``rayfronts`` on purpose: like
``multi_robot_common`` it has to be loadable (and testable) without the
package init, which pulls in every encoder, visualizer and dataset.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Topic keys, appended to the messaging service's per-robot prefix.
VOX_LAYER = "emb/voxels"
RAY_LAYER = "emb/rays"
META_LAYER = "emb/meta"
TEXT_REQUEST_LAYER = "emb/text/request"
TEXT_RESPONSE_LAYER = "emb/text/response"

# FROZEN: the planner reads these keys off emb/meta.
META_KEYS = ("k", "dim", "encoder", "lang_model", "fit_n", "cos_preservation",
             "vox_size", "query_mode")

BASIS_FILENAME = "emb_basis.pt"
DEFAULT_SAVE_DIR = "/tmp/rayfronts"

# A text request is answered synchronously on the ROS callback thread, so an
# unbounded one would stall the encoder (and with it every query) for minutes.
MAX_TEXT_PHRASES = 2000

# Pairwise cosines are O(n^2); 2000 rows is a 4M-element correlation, which is
# a few ms and plenty to tell a good basis from a bad one.
COS_SAMPLE = 2000


def _l2(x: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.normalize(x.float(), dim=-1)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
  a = a.float() - a.float().mean()
  b = b.float() - b.float().mean()
  denom = float(a.norm() * b.norm())
  if denom == 0:
    return float("nan")
  return float(torch.dot(a, b) / denom)


class EmbeddingProjector:
  """Shared D -> K projection of language-aligned features.

  Fitted once, on the union of the first non-empty map's aligned voxel
  embeddings and the aligned embeddings of a seed vocabulary, so the
  directions the planner's phrases live in survive the projection.

  Attributes:
    k: Output dimension. Clamped at fit time to min(k, D, N).
    mean: (D,) float tensor subtracted before projecting. None until fitted.
    basis: (K, D) float tensor of principal directions. None until fitted.
    fit_n: Number of rows the basis was fitted on.
    fit_cos_preservation: :meth:`cos_preservation` measured at fit time.
  """

  def __init__(self, k: int = 128):
    self.k = int(k)
    self.mean: Optional[torch.Tensor] = None
    self.basis: Optional[torch.Tensor] = None
    self.fit_n = 0
    self.fit_cos_preservation: Optional[float] = None

  def is_fitted(self) -> bool:
    return self.mean is not None and self.basis is not None

  @property
  def dim(self) -> Optional[int]:
    return None if self.basis is None else int(self.basis.shape[1])

  @torch.inference_mode()
  def fit(self, X: torch.Tensor,
          seed_text: Optional[torch.Tensor] = None) -> None:
    """Fit mean and basis on the union of map features and seed-text vectors.

    Args:
      X: (N, D) aligned, L2-normalised map features.
      seed_text: Optional (M, D) aligned, L2-normalised text features.
    """
    parts = [X.float()]
    if seed_text is not None and int(seed_text.shape[0]) > 0:
      parts.append(seed_text.float().to(X.device))
    u = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    n, d = int(u.shape[0]), int(u.shape[1])

    k = min(self.k, d, n)
    if k < self.k:
      logger.warning("[emb] k=%d is larger than the fit data allows "
                     "(n=%d, d=%d); using k=%d.", self.k, n, d, k)

    mean = u.mean(dim=0)
    centered = u - mean
    try:
      _, _, v = torch.pca_lowrank(centered, q=k, center=False)
    except Exception:                                       # noqa: BLE001
      logger.exception("[emb] pca_lowrank failed; falling back to SVD.")
      v = torch.linalg.svd(centered, full_matrices=False)[2][:k].transpose(0, 1)

    self.mean = mean
    self.basis = v.transpose(0, 1).contiguous()
    self.k = int(self.basis.shape[0])
    self.fit_n = n

  @torch.inference_mode()
  def project(self, X: torch.Tensor) -> torch.Tensor:
    """(N, D) aligned features -> (N, K) L2-normalised projected vectors."""
    if not self.is_fitted():
      raise RuntimeError("projector not fitted")
    x = X.float()
    y = (x - self.mean.to(x.device)) @ self.basis.to(x.device).transpose(0, 1)
    return _l2(y)

  @torch.inference_mode()
  def cos_preservation(self, X: torch.Tensor,
                       sample: int = COS_SAMPLE) -> float:
    """Pearson r between full-dim and projected pairwise cosines.

    Args:
      X: (N, D) aligned features. A random subset of at most ``sample`` rows
        is used so the O(n^2) cosine matrix stays small.
    """
    n = int(X.shape[0])
    if n < 3 or not self.is_fitted():
      return float("nan")
    if n > sample:
      idx = torch.randperm(n, device=X.device)[:sample]
      X = X.index_select(0, idx)
    full = _l2(X)
    proj = self.project(X)
    m = int(X.shape[0])
    iu = torch.triu_indices(m, m, offset=1, device=X.device)
    a = (full @ full.transpose(0, 1))[iu[0], iu[1]]
    b = (proj @ proj.transpose(0, 1))[iu[0], iu[1]]
    return _pearson(a, b)

  def save(self, fp: str) -> None:
    d = os.path.dirname(os.path.abspath(fp))
    if d:
      os.makedirs(d, exist_ok=True)
    torch.save(dict(
      metadata=dict(k=self.k, dim=self.dim, fit_n=self.fit_n,
                    cos_preservation=self.fit_cos_preservation),
      mean=None if self.mean is None else self.mean.cpu(),
      basis=None if self.basis is None else self.basis.cpu()), fp)

  def load(self, fp: str) -> None:
    d = torch.load(fp, map_location="cpu")
    meta = d.get("metadata", dict())
    self.mean = d["mean"]
    self.basis = d["basis"]
    self.k = int(meta.get("k", 0) or (0 if self.basis is None
                                      else self.basis.shape[0]))
    self.fit_n = int(meta.get("fit_n", 0) or 0)
    self.fit_cos_preservation = meta.get("cos_preservation")


def _cnt_column(cnt: Optional[torch.Tensor], n: int,
                like: torch.Tensor) -> torch.Tensor:
  """The mapper's hit count as a flat (N,) tensor; ones when it has none."""
  if cnt is None:
    return torch.ones(n, dtype=torch.float32, device=like.device)
  return cnt.reshape(-1)[:n].float()


def _align(mapper, encoder, feat: torch.Tensor) -> torch.Tensor:
  """Decompress -> language head -> L2 normalise, as feature_query does.

  ``align_spatial_features_with_language`` wants a (B, C, H, W) map, so the
  (N, C) map features go through it as (N, C, 1, 1) -- the same unsqueeze
  trick ``SemanticRayFrontiersMap.feature_query`` uses, which is what makes
  these embeddings comparable with the published similarity scores.
  """
  fc = getattr(mapper, "feat_compressor", None)
  if fc is not None and fc.is_fitted():
    feat = fc.decompress(feat)
  aligned = encoder.align_spatial_features_with_language(
    feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
  return _l2(aligned)


def aligned_embeddings(mapper, encoder):
  """The map's language-aligned, L2-normalised voxel and ray embeddings.

  Args:
    mapper: A semantic mapper exposing ``global_vox_*`` / ``global_rays_*``.
    encoder: A language-capable image encoder.

  Returns:
    ``(vox, rays)`` where ``vox`` is ``(vox_xyz, vox_cnt, vox_emb)`` and
    ``rays`` is ``(ray_orig_angles, ray_cnt, ray_emb)``. Either is None when
    the map holds nothing of that kind.
  """
  vox = None
  rays = None
  if mapper is None or encoder is None:
    return vox, rays

  xyz = getattr(mapper, "global_vox_xyz", None)
  feat = getattr(mapper, "global_vox_feat", None)
  if xyz is not None and feat is not None and int(xyz.shape[0]) > 0:
    n = int(xyz.shape[0])
    vox = (xyz, _cnt_column(getattr(mapper, "global_vox_cnt", None), n, xyz),
           _align(mapper, encoder, feat))

  roa = getattr(mapper, "global_rays_orig_angles", None)
  rfeat = getattr(mapper, "global_rays_feat", None)
  if roa is not None and rfeat is not None and int(roa.shape[0]) > 0:
    n = int(roa.shape[0])
    rays = (roa, _cnt_column(getattr(mapper, "global_rays_cnt", None), n, roa),
            _align(mapper, encoder, rfeat))

  return vox, rays


def embedding_fields(emb: torch.Tensor, cnt: torch.Tensor,
                     extra: Optional[Dict[str, torch.Tensor]] = None) \
    -> Dict[str, torch.Tensor]:
  """``{extra..., cnt, e_0..e_{K-1}}`` in the order the cloud fields appear."""
  fields: Dict[str, torch.Tensor] = dict()
  if extra:
    fields.update(extra)
  fields["cnt"] = cnt
  for j in range(int(emb.shape[1])):
    fields[f"e_{j}"] = emb[:, j]
  return fields


def encoder_name(encoder) -> Optional[str]:
  if encoder is None:
    return None
  for attr in ("model_version", "server_encoder"):
    v = getattr(encoder, attr, None)
    if v:
      return str(v)
  return type(encoder).__name__


def build_meta(k: int, dim: Optional[int], encoder=None, fit_n: int = 0,
               cos_preservation: Optional[float] = None,
               vox_size: Optional[float] = None,
               query_mode: str = "prompts") -> Dict[str, Any]:
  """The ``emb/meta`` payload (frozen key set, see :data:`META_KEYS`)."""
  lang_model = getattr(encoder, "lang_model", None)
  return {
    "k": int(k),
    "dim": None if dim is None else int(dim),
    "encoder": encoder_name(encoder),
    "lang_model": None if lang_model is None else str(lang_model),
    "fit_n": int(fit_n),
    "cos_preservation": (None if cos_preservation is None
                         else float(cos_preservation)),
    "vox_size": None if vox_size is None else float(vox_size),
    "query_mode": str(query_mode),
  }


def _wants(messaging_service, layer: str) -> bool:
  """Subscriber gate, mirroring the one ``voxels_sim`` publishes behind."""
  has = getattr(messaging_service, "has_subscribers", None)
  if has is None:
    return True
  try:
    return bool(has(layer))
  except Exception:                                         # noqa: BLE001
    logger.exception("[emb] subscriber check for %s failed", layer)
    return False


def publish_embeddings(messaging_service, projector: EmbeddingProjector,
                       vox=None, rays=None,
                       meta: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
  """Publish ``emb/voxels``, ``emb/rays`` and the latched ``emb/meta``.

  Args:
    messaging_service: Anything with ``publish_pc``/``publish_string``.
    projector: A fitted projector.
    vox: ``(vox_xyz, vox_cnt, vox_emb)`` from :func:`aligned_embeddings`.
    rays: ``(ray_orig_angles, ray_cnt, ray_emb)``.
    meta: When given, published once on ``emb/meta`` as latched JSON.

  Returns:
    ``{layer: point count}`` for the clouds that were actually published.
  """
  published: Dict[str, int] = dict()
  if messaging_service is None or not projector.is_fitted():
    return published

  if vox is not None and _wants(messaging_service, VOX_LAYER):
    xyz, cnt, emb = vox
    messaging_service.publish_pc(
      xyz, features=embedding_fields(projector.project(emb), cnt),
      layer=VOX_LAYER)
    published[VOX_LAYER] = int(xyz.shape[0])

  if rays is not None and _wants(messaging_service, RAY_LAYER):
    roa, cnt, emb = rays
    extra = dict(theta=roa[:, 3], phi=roa[:, 4])
    messaging_service.publish_pc(
      roa[:, :3], features=embedding_fields(projector.project(emb), cnt,
                                            extra=extra),
      layer=RAY_LAYER)
    published[RAY_LAYER] = int(roa.shape[0])

  if meta is not None:
    messaging_service.publish_string(META_LAYER, json.dumps(meta),
                                     latched=True)
  return published


def _cfg_get(cfg, key: str, default=None):
  """``cfg[key]`` for a DictConfig, ``cfg.key`` otherwise, default for null."""
  if cfg is None:
    return default
  value = None
  try:
    if key in cfg:
      value = cfg[key]
  except TypeError:
    value = getattr(cfg, key, None)
  return default if value is None else value


def default_save_dir() -> str:
  """``$LOG_DIR`` / the hydra run dir if there is one, else /tmp/rayfronts."""
  for env_key in ("RAYFRONTS_LOG_DIR", "LOG_DIR"):
    d = os.environ.get(env_key)
    if d and os.path.isdir(d):
      return d
  try:
    from hydra.core.hydra_config import HydraConfig
    if HydraConfig.initialized():
      d = str(HydraConfig.get().runtime.output_dir)
      if os.path.isdir(d):
        return d
  except Exception:                                         # noqa: BLE001
    pass
  return DEFAULT_SAVE_DIR


def read_seed_vocab(path: Optional[str]) -> List[str]:
  """One phrase per line, blanks and ``#`` comments dropped, order kept."""
  if not path:
    return []
  try:
    with open(path, "r", encoding="UTF-8") as f:
      lines = [l.strip() for l in f.readlines()]
  except OSError:
    logger.exception("[emb] could not read seed vocabulary %s", path)
    return []
  out = [l for l in lines if l and not l.startswith("#")]
  return list(dict.fromkeys(out))


class EmbeddingExporter:
  """Owns the basis, the periodic publish and the text-embedding service.

  Every public entry point swallows its own exceptions: this runs inside the
  mapping loop and a bad projection must degrade the planner's input, never
  stop the map.

  Attributes:
    enabled: False (the default) means nothing is ever created or published.
    period: Publish every this many frames.
    projector: The shared :class:`EmbeddingProjector`.
  """

  def __init__(self, cfg, mapper=None, encoder=None, messaging_service=None):
    emb = _cfg_get(cfg, "emb", None)
    querying = _cfg_get(cfg, "querying", None)

    self.cfg = cfg
    self.mapper = mapper
    self.encoder = encoder
    self.messaging_service = messaging_service

    self.enabled = bool(_cfg_get(emb, "enabled", False))
    self.k = int(_cfg_get(emb, "k", 128))
    period = _cfg_get(emb, "period", None)
    self.period = int(_cfg_get(querying, "period", 10)
                      if period is None else period)
    self.seed_vocab = _cfg_get(emb, "seed_vocab", None)
    self.fit_min_voxels = int(_cfg_get(emb, "fit_min_voxels", 500))
    self.save_dir = _cfg_get(emb, "save_dir", None) or default_save_dir()
    self.query_mode = str(_cfg_get(querying, "text_query_mode", None)
                          or "prompts")

    self.projector = EmbeddingProjector(self.k)
    self._meta: Optional[Dict[str, Any]] = None
    self._lock = threading.RLock()
    self._text_lock = threading.RLock()
    self._started = False

  # ------------------------------------------------------------------ #
  # Lifecycle
  # ------------------------------------------------------------------ #

  def start(self) -> None:
    """Subscribe to ``emb/text/request``. No-op while disabled."""
    if not self.enabled or self._started or self.messaging_service is None:
      return
    subscribe = getattr(self.messaging_service, "subscribe_string", None)
    if subscribe is None:
      logger.warning("[emb] messaging service %s cannot subscribe to %s; "
                     "text embedding requests are unavailable.",
                     type(self.messaging_service).__name__,
                     TEXT_REQUEST_LAYER)
    else:
      try:
        subscribe(TEXT_REQUEST_LAYER, self.on_text_request)
      except Exception:                                     # noqa: BLE001
        logger.exception("[emb] could not subscribe to %s", TEXT_REQUEST_LAYER)
    self._started = True
    logger.info("[emb] embedding export enabled: k=%d period=%d "
                "fit_min_voxels=%d seed_vocab=%s save_dir=%s",
                self.k, self.period, self.fit_min_voxels, self.seed_vocab,
                self.save_dir)

  def status_fields(self) -> Optional[Dict[str, Any]]:
    """The ``emb`` block of the status JSON, or None while disabled."""
    if not self.enabled:
      return None
    return dict(k=int(self.projector.k), fit_n=int(self.projector.fit_n),
                cos_preservation=self.projector.fit_cos_preservation)

  # ------------------------------------------------------------------ #
  # Publishing
  # ------------------------------------------------------------------ #

  def maybe_run(self, frame_idx: int) -> None:
    """Publish if this frame is a publish frame. Never raises."""
    if not self.enabled or self.period <= 0:
      return
    if int(frame_idx) % self.period != 0:
      return
    try:
      self.run_once()
    except Exception:                                       # noqa: BLE001
      logger.exception("[emb] embedding export failed; continuing.")

  def run_once(self) -> Dict[str, int]:
    """Align, fit (once), project and publish the current map."""
    if not self.enabled or self.messaging_service is None:
      return dict()
    with self._lock:
      if not self._any_subscriber():
        return dict()
      vox, rays = aligned_embeddings(self.mapper, self.encoder)
      if vox is None and rays is None:
        return dict()
      if not self.projector.is_fitted():
        if vox is None or int(vox[2].shape[0]) < self.fit_min_voxels:
          return dict()
        self._fit(vox[2])
      published = publish_embeddings(self.messaging_service, self.projector,
                                     vox=vox, rays=rays, meta=self._meta)
      self._meta = None
      return published

  def _any_subscriber(self) -> bool:
    """Skip the (expensive) language head when nothing wants the result.

    Text requests arriving before anything subscribes to a cloud are answered
    with "projector not fitted" rather than paying for a fit nobody reads.
    """
    return (_wants(self.messaging_service, VOX_LAYER) or
            _wants(self.messaging_service, RAY_LAYER))

  def _fit(self, vox_emb: torch.Tensor) -> None:
    seed = self._encode_seed_vocab(vox_emb)
    self.projector.fit(vox_emb, seed_text=seed)
    self.projector.fit_cos_preservation = \
      self.projector.cos_preservation(vox_emb)
    logger.info("[emb] basis fitted: k=%d fit_n=%d (%d voxels + %d seed "
                "phrases) cos_preservation=%.4f", self.projector.k,
                self.projector.fit_n, int(vox_emb.shape[0]),
                0 if seed is None else int(seed.shape[0]),
                float(self.projector.fit_cos_preservation))

    self._meta = build_meta(
      k=self.projector.k, dim=self.projector.dim, encoder=self.encoder,
      fit_n=self.projector.fit_n,
      cos_preservation=self.projector.fit_cos_preservation,
      vox_size=getattr(self.mapper, "vox_size", None),
      query_mode=self.query_mode)

    path = os.path.join(self.save_dir, BASIS_FILENAME)
    try:
      self.projector.save(path)
      logger.info("[emb] basis saved to %s", path)
    except Exception:                                       # noqa: BLE001
      logger.exception("[emb] could not save the basis to %s", path)

  @torch.inference_mode()
  def _encode_seed_vocab(self, like: torch.Tensor) -> Optional[torch.Tensor]:
    """Aligned seed-vocabulary vectors, so text directions survive the PCA."""
    phrases = read_seed_vocab(self.seed_vocab)
    if not phrases:
      logger.warning("[emb] no seed vocabulary (emb.seed_vocab=%s); the basis "
                     "is fitted on map features alone, which can drop the "
                     "directions the planner's phrases live in.",
                     self.seed_vocab)
      return None
    try:
      feats = self._encode_text(phrases, self.query_mode)
    except Exception:                                       # noqa: BLE001
      logger.exception("[emb] could not encode the seed vocabulary %s",
                       self.seed_vocab)
      return None
    return _l2(feats).to(like.device)

  def _encode_text(self, phrases: Sequence[str], mode: str) -> torch.Tensor:
    if self.encoder is None:
      raise RuntimeError("no encoder")
    if mode == "labels":
      return self.encoder.encode_labels(list(phrases))
    if mode == "prompts":
      return self.encoder.encode_prompts(list(phrases))
    raise ValueError(f"unknown mode {mode!r}")

  # ------------------------------------------------------------------ #
  # Text embedding service
  # ------------------------------------------------------------------ #

  def on_text_request(self, payload) -> None:
    """Answer one ``emb/text/request``. Never raises."""
    if not self.enabled or self.messaging_service is None:
      return
    try:
      response = self.text_response(payload)
    except Exception:                                       # noqa: BLE001
      logger.exception("[emb] text request failed")
      return
    if response is None:
      return
    try:
      self.messaging_service.publish_string(TEXT_RESPONSE_LAYER, response)
    except Exception:                                       # noqa: BLE001
      logger.exception("[emb] could not publish the text response")

  def text_response(self, payload) -> Optional[str]:
    """The JSON answer for one request payload.

    Answered in order under one lock -- the planner matches responses by
    ``id`` but relies on the ordering to pipeline its vocabulary.
    """
    with self._text_lock:
      req = payload
      if isinstance(req, (bytes, bytearray)):
        req = req.decode("utf-8", "replace")
      if isinstance(req, str):
        try:
          req = json.loads(req)
        except ValueError:
          return _error(None, "malformed json")
      if not isinstance(req, dict):
        return _error(None, "malformed request")

      req_id = req.get("id")
      phrases = req.get("phrases")
      if not isinstance(phrases, (list, tuple)) or len(phrases) == 0:
        return _error(req_id, "no phrases")
      if len(phrases) > MAX_TEXT_PHRASES:
        return _error(req_id, f"too many phrases: {len(phrases)} > "
                              f"{MAX_TEXT_PHRASES}")
      phrases = [str(p) for p in phrases]

      mode = req.get("mode") or "prompts"
      if mode not in ("prompts", "labels"):
        return _error(req_id, f"unknown mode {mode!r}")

      if not self.projector.is_fitted():
        return _error(req_id, "projector not fitted")

      try:
        feats = self._encode_text(phrases, mode)
        vectors = self.projector.project(_l2(feats))
      except Exception:                                     # noqa: BLE001
        logger.exception("[emb] could not encode %d phrase(s)", len(phrases))
        return _error(req_id, "encoding failed")

      return json.dumps({
        "id": req_id,
        "k": int(vectors.shape[1]),
        "phrases": phrases,
        "vectors": vectors.cpu().tolist(),
      })


def _error(req_id, message: str) -> str:
  logger.warning("[emb] text request %r rejected: %s", req_id, message)
  return json.dumps({"id": req_id, "error": message})
