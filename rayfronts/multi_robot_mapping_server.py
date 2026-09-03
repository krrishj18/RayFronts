"""ONE semantic map, many robots.

    python3 -m rayfronts.multi_robot_mapping_server --config-name shared_humans \\
            dataset.robot_ids=[1,2] encoder=client \\
            encoder.socket=/tmp/rayfronts/encoder.sock

Everything structural is inherited from :class:`rayfronts.mapping_server
.MappingServer`; what this subclass adds is the multi-robot plumbing:

* it iterates :class:`~rayfronts.datasets.MultiRobotRos2Subscriber` directly
  instead of through a ``DataLoader`` -- frames from different robots must not
  be batched together, and each frame carries the robot that produced it;
* it tells the visualiser which robot is speaking before logging input layers;
* it merges every robot's ``new_text_query`` and ``guiding_queries`` into one
  query column set, and ports the original RAVEN ``delete_queries`` so a
  guiding label dropped by every robot leaves the map again (never a label that
  came in through ``new_text_query``);
* it publishes ``/robot_i/rayfronts/status`` at 1 Hz, which is what
  ``semantic_search_task`` waits on in shared mode;
* it logs OPEN-SET PROBES (``RAYFRONTS_PROBE_PROMPTS`` /
  ``RAYFRONTS_PROBE_POINTS``) — alternative query wordings scored against the
  map, and the raw scores at known ground-truth casualty positions — without
  either of them ever entering the published query set.

The mapper itself has no internal locking (see the internals report), so
``process_posed_rgbd`` / ``feature_query`` / ``vis_*`` all stay on this one
thread; the ROS callback threads only ever touch the query bookkeeping, under
``_query_lock``.
"""

import atexit
import json
import logging
import os
import random
import signal
import threading
import time
from functools import partial

import hydra
import numpy as np
import torch

from rayfronts.mapping_server import MappingServer
from rayfronts import multi_robot_common as mrc

logger = logging.getLogger(__name__)


def _env_or_cfg(cfg, env_key: str, cfg_key: str):
  """Env var if it is set and non-blank, else ``cfg.<cfg_key>``, else None.

  Env wins because these arrive from an OSMO mission's ``env:`` block, which
  reaches the mapping server through offboard-compute's container environment
  (``offboard_compute.sh`` starts it with a plain ``python3 -m``, so the
  process inherits it); the config key exists so a bench run can set the same
  thing without a container.
  """
  val = os.environ.get(env_key)
  if val is not None and str(val).strip():
    return val
  try:
    if hasattr(cfg, "get"):
      return cfg.get(cfg_key, None)
    return getattr(cfg, cfg_key, None)
  except Exception:                                         # noqa: BLE001
    return None


class MultiRobotMappingServer(MappingServer):
  """MappingServer driven by several robots on several ROS domains.

  Attributes:
    registry: GuidingQueryRegistry tracking who wants which guiding label.
    status_period_s: How often the per-robot status topic is published.
  """

  # ---- open-set probe state (diagnostics; NEVER published) ----------------
  # Class-level defaults so a partially constructed server (the unit tests
  # build one with ``__new__``) and the logging path below can always read
  # them with a plain attribute access.
  _probe_labels = ()
  _probe_feats = None
  _probe_points_flu = ()

  # How close a semantic voxel has to be to a probe point to count as "the map
  # at that position". 3 m is 6 voxels at the shared_humans vox_size of 0.5 —
  # loose enough that a GT position recorded at the casualty's centroid still
  # finds the voxel the drone actually carved, tight enough that a hit is
  # about THAT body and not the building behind it.
  PROBE_RADIUS_M = 3.0

  # The softmax temperature the LIVE pipeline uses: rayfronts/utils.py
  # ``compute_cos_sim`` returns ``torch.softmax(100 * sim, dim=-1)``. The solo
  # score below has to use the same 100 or it cannot be compared against the
  # mission's voxel_score_threshold.
  PROBE_SOFTMAX_TEMP = 100.0

  # Labels that form the CONTRAST set for the solo softmax (see
  # ``_solo_probs``). Resolved at init from RAYFRONTS_PROBE_BACKGROUND, else
  # from whatever querying.query_file seeded at startup.
  _probe_background = frozenset()

  def __init__(self, cfg):
    super().__init__(cfg)

    self.registry = mrc.GuidingQueryRegistry()
    # Whatever the query file seeded (the background vocabulary) plus anything
    # the config added is "pinned": it is the mission vocabulary and must never
    # be deleted by the guiding-query logic.
    with self._query_lock:
      seeded = list(self._queries_labels_history)
    if seeded:
      self.registry.pin(sorted(seeded))
      logger.info("Pinned %d background/query-file labels: %s",
                  len(seeded), sorted(seeded))

    self.status_period_s = float(
      getattr(cfg, "status_period_s", 1.0) or 1.0)

    self._init_probes(cfg, seeded)

    ds = self.dataset
    ms = self.messaging_service
    if ms is None:
      raise ValueError(
        "multi_robot_mapping_server needs a messaging_service "
        "(messaging_service=multi_ros).")
    if not hasattr(ds, "streams"):
      raise ValueError(
        "multi_robot_mapping_server needs dataset=multi_ros2isaacsim "
        f"(MultiRobotRos2Subscriber), got {type(ds).__name__}.")

    # Rewire the callbacks MappingServer set up for the single-robot case.
    ms.text_query_callback = self._on_new_text_query
    ms.guiding_callback = self._on_guiding_queries
    if hasattr(ms, "set_anchor_source"):
      ms.set_anchor_source(ds)
    if self.vis is not None and hasattr(self.vis, "set_anchor_source"):
      self.vis.set_anchor_source(ds)

    self._status_stop = threading.Event()
    self._status_thread = threading.Thread(
      target=self._status_loop, name="rayfronts_status", daemon=True)
    self._status_thread.start()

  # ------------------------------------------------------------------ #
  # Query set: union over robots, OG delete semantics
  # ------------------------------------------------------------------ #

  def _on_new_text_query(self, label):
    """A robot published a label — or a whole vocabulary — on new_text_query.

    A JSON array in one String message is the ATOMIC form (added 2026-09-02):
    semantic_search_task sends the full ordered vocabulary as one message so
    a mapper that (re)starts at any moment receives the entire list in order
    in a single delivery — no burst of per-label messages whose head can be
    dropped by the DDS discovery race or split by a restart, which is how
    `road` ended up as q0/sim_0 and rendered every road as `person`.
    Plain single-label strings keep working (the legacy per-robot form).
    """
    if isinstance(label, (list, tuple)):
      labels = [str(x) for x in label]
    else:
      text = str(label)
      labels = None
      if text.lstrip().startswith("["):
        try:
          parsed = json.loads(text)
          if isinstance(parsed, list):
            labels = [str(x) for x in parsed]
        except (ValueError, TypeError):
          labels = None
      if labels is None:
        labels = [text]
    self.registry.pin(labels)
    self.add_queries(labels)

  def _on_guiding_queries(self, robot_id, labels):
    """A robot published its CURRENT guiding list; recompute the union."""
    added, removed = self.registry.set_guiding(robot_id, labels)
    if added:
      logger.info("[%s] guiding adds %s", mrc.robot_name(robot_id), added)
      self.add_queries(added)
    if removed:
      logger.info("[%s] guiding drops %s (no robot lists them any more)",
                  mrc.robot_name(robot_id), removed)
      self.delete_queries(removed)

  @torch.inference_mode()
  def delete_queries(self, labels):
    """Remove query columns. Port of RAVEN's ``MappingServer.delete_queries``.

    Only labels that are (a) currently columns, (b) not pinned and (c) not
    referenced by any robot's guiding list are removed. Both the label list and
    the matching rows of the feature matrix are dropped, and the label leaves
    ``_queries_labels_history`` so it can come back later.
    """
    if isinstance(labels, str):
      labels = [labels]
    with self._query_lock:
      if self._queries_labels is None or self._queries_feats is None:
        return
      current = self._queries_labels.get("text")
      feats = self._queries_feats.get("text")
      if not current or feats is None:
        return

      victims = [l for l in dict.fromkeys(labels)
                 if l in current and self.registry.deletable(l)]
      if not victims:
        return
      victim_set = set(victims)

      keep_idx = [i for i, l in enumerate(current) if l not in victim_set]
      deleted = [l for l in current if l in victim_set]

      self._queries_labels["text"] = [current[i] for i in keep_idx]
      mask = torch.zeros(len(current), dtype=torch.bool, device=feats.device)
      if keep_idx:
        mask[torch.tensor(keep_idx, device=feats.device)] = True
      self._queries_feats["text"] = feats[mask]
      self._queries_labels_history.difference_update(deleted)
      self._queries_updated = True
    logger.info("Deleted query columns %s. Remaining: %s",
                deleted, self._queries_labels["text"])

  # ------------------------------------------------------------------ #
  # Open-set score logging (raw cosines, no softmax)
  # ------------------------------------------------------------------ #

  _RAW_LOG_EVERY = 2   # run_queries fires every querying.period frames; x2

  # ------------------------------------------------------------------ #
  # Open-set PROBES: alternative wordings and ground-truth positions
  # ------------------------------------------------------------------ #

  def _init_probes(self, cfg, seeded_labels=()):
    """Register ``RAYFRONTS_PROBE_PROMPTS`` / ``RAYFRONTS_PROBE_POINTS``.

    Neither ever reaches ``_queries_labels`` / ``_queries_feats``: with
    ``querying.compute_prob=True`` the published score of every label is a
    softmax ACROSS the column set, so adding "casualty" as a column to see how
    it scores would move the very numbers raven thresholds on. The probe feats
    live in their own tensor and are only ever stacked onto the query feats
    for the read-only ``feature_query`` the logger makes.

    ``seeded_labels`` is what ``querying.query_file`` pinned at startup — the
    background vocabulary, and the default contrast set for the solo softmax.
    """
    self._probe_labels = ()
    self._probe_feats = None
    self._probe_points_flu = ()

    # The solo softmax needs to know WHICH live labels are background. There
    # is no runtime signal for it: raven publishes target and background
    # together on new_text_query and add_queries() de-dupes through a set, so
    # column order does not even preserve "target first". Hence:
    #   1. RAYFRONTS_PROBE_BACKGROUND / probe_background — explicit, and the
    #      only option that works with the shipped querying.query_file: null;
    #   2. else whatever the query file seeded at startup (the "Pinned N
    #      background/query-file labels" set);
    #   3. else nothing — solo is undefined and only raw cosines are logged.
    background = mrc.parse_probe_prompts(
      _env_or_cfg(cfg, "RAYFRONTS_PROBE_BACKGROUND", "probe_background"))
    source = "RAYFRONTS_PROBE_BACKGROUND"
    if not background:
      background = sorted(str(x) for x in (seeded_labels or ()))
      source = "querying.query_file"
    self._probe_background = frozenset(background)
    if background:
      logger.info("[probe] solo-softmax contrast set (%d labels, from %s): "
                  "%s", len(background), source, background)

    points = mrc.parse_probe_points(
      _env_or_cfg(cfg, "RAYFRONTS_PROBE_POINTS", "probe_points"))
    if points:
      self._probe_points_flu = tuple(points)
      logger.info("[probe] %d probe POINT(s) registered (world FLU, matched "
                  "to the nearest semantic voxel within %.1f m): %s",
                  len(points), self.PROBE_RADIUS_M,
                  ", ".join("(" + ",".join(f"{v:.2f}" for v in p) + ")"
                            for p in points))

    prompts = mrc.parse_probe_prompts(
      _env_or_cfg(cfg, "RAYFRONTS_PROBE_PROMPTS", "probe_prompts"))
    if prompts:
      self._encode_probe_prompts(prompts)

    if not points and not prompts:
      logger.info("[probe] no RAYFRONTS_PROBE_PROMPTS / RAYFRONTS_PROBE_"
                  "POINTS set; open-set probe logging is off.")
    elif not background:
      logger.warning(
        "[probe] no contrast set: set RAYFRONTS_PROBE_BACKGROUND to the "
        "mission's background_queries (the shipped config has "
        "querying.query_file: null, so nothing is seeded at startup). Raw "
        "cosines will still be logged; the solo softmax will not.")

  @torch.inference_mode()
  def _encode_probe_prompts(self, prompts):
    """Encode the probe texts ONCE, exactly as ``add_queries`` would."""
    encoder = getattr(self, "encoder", None)
    if encoder is None or not hasattr(encoder, "encode_labels"):
      logger.warning("[probe] %d probe prompt(s) requested but this server has "
                     "no text-capable encoder; probes disabled.", len(prompts))
      return

    mode = self.cfg.querying.text_query_mode
    try:
      if mode == "labels":
        feats = encoder.encode_labels(list(prompts))
      elif mode == "prompts":
        feats = encoder.encode_prompts(list(prompts))
      else:
        raise ValueError(f"Invalid querying.text_query_mode {mode!r}")
    except Exception:                                       # noqa: BLE001
      logger.exception("[probe] failed to encode probe prompts %s; probes "
                       "disabled.", list(prompts))
      return

    # Probe feats are stored RAW, exactly like the query feats (add_queries
    # stores raw since the fit-on-queries fix): _log_raw_query_scores
    # compresses the stacked [vocab; probes] tensor lazily once the first
    # frame has fitted the compressor. NEVER fit the compressor here — a
    # basis fitted to five diagnostic prompts would corrupt the whole map.

    self._probe_labels = tuple(str(p) for p in prompts)
    self._probe_feats = feats
    logger.info("[probe] %d probe PROMPT(s) encoded via %s and held OUT of "
                "the published query set: %s", len(self._probe_labels),
                mode, list(self._probe_labels))

  def _probe_row_sets(self, labels, n_probe):
    """(background rows, rows that get a solo score) into the stacked sim.

    Rows ``0..n_query-1`` are the live vocabulary; ``n_query..`` are the probe
    prompts. Background rows are the contrast set; everything else — the live
    POSITIVE(s) and every probe prompt — gets a solo score.
    """
    n_query = len(labels)
    bg = [i for i, l in enumerate(labels) if str(l) in self._probe_background]
    bg_set = set(bg)
    solo = [i for i in range(n_query) if i not in bg_set]
    solo += [n_query + j for j in range(n_probe)]
    return bg, solo

  def _solo_probs(self, sim, k, solo_rows, bg_rows):
    """Softmax of each solo row against ONLY the background rows, at voxel k.

    ``solo_i = exp(T*cos_i) / (exp(T*cos_i) + sum_b exp(T*cos_b))`` where b
    runs over the background labels alone. This is what the live score of that
    wording WOULD be if it replaced "person" as the sole positive query
    against the same background vocabulary — directly comparable to the
    mission's ``voxel_score_threshold``.

    The probe prompts are near-synonyms ("casualty", "human body", "person
    lying down"). A joint softmax over all of them splits the probability mass
    between them and every one reads as a miss, which is exactly the lie this
    avoids: siblings are NOT in each other's denominators, and neither is the
    live positive. Each solo score is therefore independent of which other
    probes happen to be configured.

    SANITY CHECK the operator can use: when the live vocabulary is exactly one
    positive plus this contrast set (the mission's ``person`` + 6
    ``background_queries``), the positive's solo score is arithmetically
    IDENTICAL to its published softmax — same rows, same temperature. If
    ``person``'s solo here disagrees with ``sim_0`` on the published cloud,
    the contrast set does not match the mission's background_queries.
    """
    if not bg_rows or not solo_rows:
      return {}
    t = float(self.PROBE_SOFTMAX_TEMP)
    bg = sim[bg_rows, k].float().reshape(-1) * t
    out = {}
    for r in solo_rows:
      if r >= sim.shape[0]:
        continue
      z = torch.cat((sim[r, k].float().reshape(1) * t, bg))
      out[r] = float(torch.softmax(z, dim=0)[0])
    return out

  @staticmethod
  def _probe_kv(labels, sim, k, offset=0, solo=None, only_solo=False,
                width=40):
    """``label=raw`` (or ``label=raw/solo``) pairs for voxel ``k``.

    Spaces become underscores: "person lying down=0.241" is not parseable,
    "person_lying_down=0.241" is, and every consumer of this log (grep, the
    runbook's awk one-liners) splits on whitespace. ``raw`` is the cosine;
    ``solo`` is :meth:`_solo_probs` — present for the live positive(s) and the
    probe prompts, absent for the background labels (which ARE the
    denominator, so a solo score for them would be meaningless).

    ``only_solo`` drops the rows that have no solo score, which is how the
    ``[probe top-*]`` line stays short: there the point is the comparison
    "person 0.19/0.43 vs casualty 0.24/0.71", not the whole background.
    """
    out = []
    for i, lab in enumerate(labels):
      row = offset + i
      if row >= sim.shape[0]:
        break
      s = None if solo is None else solo.get(row)
      if only_solo and s is None:
        continue
      name = str(lab)[:width].strip().replace(" ", "_")
      val = f"{float(sim[row, k]):.3f}"
      if s is not None:
        val = f"{val}/{s:.3f}"
      out.append(f"{name}={val}")
    return " ".join(out)

  def _nearest_vox(self, vox_xyz, point_flu):
    """(index, distance) of the map voxel nearest a world-FLU probe point.

    Torch mirror of :func:`multi_robot_common.nearest_index` (kept in torch so
    a million-voxel map is not copied to the host every log cycle);
    ``tests/test_multi_mapping_server.py`` asserts the two agree.
    """
    axes = mrc.probe_axes(point_flu)
    target = mrc.world_flu_to_rdf(point_flu)
    cols = torch.tensor(list(axes), dtype=torch.long, device=vox_xyz.device)
    tgt = torch.as_tensor([float(target[a]) for a in axes],
                          dtype=torch.float32, device=vox_xyz.device)
    d = torch.linalg.norm(
      vox_xyz.index_select(1, cols).float() - tgt, dim=1)
    k = int(torch.argmin(d))
    return k, float(d[k])

  @torch.inference_mode()
  def _log_probe_scores(self, r, labels, probe_labels, n_query,
                        bg_rows=(), solo_rows=()):
    """The two probe log blocks. Purely read-only over ``r``.

    (a) one pair of rows per probe POINT — what the live vocabulary and the
        probe wordings score at the known GT casualty position;
    (b) is emitted inline by ``_log_raw_query_scores`` next to ``[raw top]``.
    """
    points = tuple(getattr(self, "_probe_points_flu", ()) or ())
    if not points:
      return
    sim = r.get("vox_sim")
    xyz = r.get("vox_xyz")
    if xyz is None:
      xyz = getattr(self.mapper, "global_vox_xyz", None)
    if sim is None or xyz is None or sim.numel() == 0:
      return
    n = int(xyz.shape[0])
    if int(sim.shape[1]) != n:
      # feature_query and global_vox_xyz disagreed (a concurrent prune) —
      # indexing would be meaningless, so skip this cycle rather than lie.
      return

    bg_rows = list(bg_rows)
    solo_rows = list(solo_rows)
    for p in points:
      tag = "(" + ",".join(f"{v:.2f}" for v in p) + ")"
      k, dist = self._nearest_vox(xyz, p)
      if k is None or dist > self.PROBE_RADIUS_M:
        logger.info("[probe@%s] no voxel within %.1f m (nearest %.2f m of "
                    "%d voxels)", tag, self.PROBE_RADIUS_M, dist, n)
        continue
      vx, vy, vz = (float(xyz[k, 2]), float(-xyz[k, 0]), float(-xyz[k, 1]))
      solo = self._solo_probs(sim, k, solo_rows, bg_rows)
      logger.info("[probe@%s] vox#%d d=%.2fm @(%.1f,%.1f,%.1f) labels: %s",
                  tag, k, dist, vx, vy, vz,
                  self._probe_kv(labels[:n_query], sim, k, solo=solo))
      if probe_labels:
        logger.info("[probe@%s] probes: %s", tag,
                    self._probe_kv(probe_labels, sim, k, offset=n_query,
                                   solo=solo))

  def run_queries(self):
    super().run_queries()
    self._raw_log_cnt = getattr(self, "_raw_log_cnt", 0) + 1
    if self._raw_log_cnt % self._RAW_LOG_EVERY == 0:
      self._log_raw_query_scores()

  @torch.inference_mode()
  def _log_raw_query_scores(self):
    """Log per-label RAW cosine stats alongside the softmax pipeline.

    The published voxels_sim/rays_sim carry softmax over the query set
    (compute_prob=True), which is what the planner thresholds — but softmax
    hides the ABSOLUTE person-likeness of a voxel, so a false positive is
    indistinguishable from a weak-vocabulary artefact in the logs. One line
    per label with the raw max / p99 / mean makes the open-set picture
    readable straight from this log (requested live 2026-09-02: "log the
    rayfronts score before softmax so we know the open-set classification").

    Probe prompts ride along on the SAME ``feature_query`` call — stacked
    BELOW the vocabulary rows — so their scores are guaranteed to index the
    same voxel array as the vocabulary's, and the map's features are aligned
    to language once per cycle instead of twice.
    """
    with self._query_lock:
      feats = (None if self._queries_feats is None
               else self._queries_feats.get("text"))
      labels = (list(self._queries_labels.get("text") or [])
                if self._queries_labels else [])
    if feats is None or not labels:
      return
    n_query = len(labels)

    probe_feats = getattr(self, "_probe_feats", None)
    probe_labels = list(getattr(self, "_probe_labels", ()) or ())
    query_feats = feats
    if probe_feats is not None and int(probe_feats.shape[0]) > 0:
      try:
        query_feats = torch.cat(
          (feats, probe_feats.to(device=feats.device, dtype=feats.dtype)),
          dim=0)
      except Exception:                                   # noqa: BLE001
        logger.exception("[probe] could not stack the probe feats onto the "
                         "query feats; skipping probes this cycle")
        probe_labels = []
    else:
      probe_labels = []

    # Which stacked rows are the contrast set, and which get a solo softmax.
    bg_rows, solo_rows = self._probe_row_sets(labels, len(probe_labels))

    # Stored feats are RAW (add_queries / _encode_probe_prompts). Mirror
    # run_queries' lazy compression so these cosines live in the SAME feature
    # space the published (thresholded) scores are computed in.
    compressed_q = bool(self.cfg.querying.compressed)
    fc = getattr(self, "feat_compressor", None)
    if fc is not None and compressed_q:
      if fc.is_fitted():
        query_feats = fc.compress(query_feats)
      else:
        compressed_q = False
    try:
      r = self.mapper.feature_query(
        query_feats, softmax=False, compressed=compressed_q)
    except torch.OutOfMemoryError:
      # Do NOT swallow: the supervisor restart (offboard_compute.sh) is the
      # recovery path. A swallowed OOM here is how the 2026-09-02 run limped
      # for 8 minutes publishing nothing while raven oscillated on stale
      # frontiers.
      raise
    except Exception:                                     # noqa: BLE001
      logger.exception("raw-score query failed")
      return
    if not isinstance(r, dict):
      return
    bg_set = set(bg_rows)
    for key, tag in (("vox_sim", "vox"), ("ray_sim", "ray")):
      sim = r.get(key)
      if sim is None or sim.numel() == 0:
        continue
      probe_top_seen = set()
      # sim is [Q, N] (feature_query transposes) — one row per query label.
      for qi, lab in enumerate(labels[:sim.shape[0]]):
        row = sim[qi].float()
        n = int(row.numel())
        if n == 0:
          continue
        p99 = torch.quantile(row, 0.99) if n > 1 else row.max()
        logger.info(
          "[raw %s] %-14s n=%-7d max=%.3f p99=%.3f mean=%.3f >0.5:%d >0.6:%d",
          tag, lab, n, float(row.max()), float(p99), float(row.mean()),
          int((row > 0.5).sum()), int((row > 0.6).sum()))
        # CONFUSION ROW for the argmax element: where the most label-like
        # thing in the map is, and what ELSE it resembles in absolute raw
        # terms — the line that explains a false positive ("the 'person' at
        # (x,y) is really wood-debris-shaped: person=0.19 vs debris=0.23").
        if tag == "vox" and n > 0:
          try:
            k = int(torch.argmax(row))
            xyz = getattr(self.mapper, "global_vox_xyz", None)
            if xyz is not None and xyz.shape[0] == n:
              # map stores RDF; print world FLU (x=z, y=-x, z=-y).
              px, py, pz = (float(xyz[k, 2]), float(-xyz[k, 0]),
                            float(-xyz[k, 1]))
              confus = " ".join(
                f"{labels[qj][:12]}={float(sim[qj, k]):.3f}"
                for qj in range(min(len(labels), sim.shape[0])))
              logger.info("[raw top] %-14s @(%.1f,%.1f,%.1f): %s",
                          lab, px, py, pz, confus)
              # (b) THE SAME VOXEL, scored by the probe wordings. For the
              # `person` column this is the "[probe top-person]" row: it says
              # whether the map's most person-like voxel is better described
              # as "casualty" / "mannequin" / "pile of debris" — i.e. whether
              # a different query text would find what `person` is missing.
              #
              # Only for the live POSITIVE columns, and once per voxel: a
              # background label's argmax is not a thing we are probing for,
              # and with 7 labels most of them share one argmax voxel, so
              # without this the same line is logged six times a cycle.
              if (probe_labels and qi not in bg_set
                  and k not in probe_top_seen):
                probe_top_seen.add(k)
                solo = self._solo_probs(sim, k, solo_rows, bg_rows)
                # The live positive(s) FIRST as the reference column, then the
                # probes: "person=0.19/0.43 casualty=0.24/0.71" reads straight
                # off as "this wording would cross the 0.6 gate, person does
                # not". Background labels are dropped (they are the
                # denominator); the full picture is on the [raw top] line.
                logger.info(
                  "[probe top-%s] @(%.1f,%.1f,%.1f): %s %s",
                  lab, px, py, pz,
                  self._probe_kv(labels[:n_query], sim, k, solo=solo,
                                 only_solo=True),
                  self._probe_kv(probe_labels, sim, k, offset=n_query,
                                 solo=solo))
          except Exception:                             # noqa: BLE001
            pass

    # (a) The probe POINTS: what the map holds at the known GT positions.
    try:
      self._log_probe_scores(r, labels, probe_labels, n_query,
                             bg_rows, solo_rows)
    except Exception:                                     # noqa: BLE001
      logger.exception("[probe] probe-point logging failed")

  def current_query_labels(self):
    """Query labels in column order (what the ``q{k}_{label}`` topics use)."""
    with self._query_lock:
      if not self._queries_labels:
        return []
      out = []
      for k in ("text", "img"):
        v = self._queries_labels.get(k)
        if v:
          out.extend(list(v))
      return out

  # ------------------------------------------------------------------ #
  # Status topic
  # ------------------------------------------------------------------ #

  def _map_counts(self):
    m = self.mapper
    vox = getattr(m, "global_vox_xyz", None)
    rays = getattr(m, "global_rays_orig_angles", None)
    try:
      n_vox = 0 if vox is None else int(vox.shape[0])
    except Exception:
      n_vox = 0
    try:
      n_ray = 0 if rays is None else int(rays.shape[0])
    except Exception:
      n_ray = 0
    return n_vox, n_ray

  def publish_status_once(self):
    ds = self.dataset
    ms = self.messaging_service
    n_vox, n_ray = self._map_counts()
    labels = self.current_query_labels()
    ts = time.time()
    for rid in ds.robot_ids:
      anchored = ds.is_anchored(rid)
      boot = ds.boot_enu(rid)
      status = mrc.build_status(
        robot_id=rid, domain_id=ds.domain_of(rid), anchored=anchored,
        boot_enu=(list(boot) if anchored else None),
        frames_robot=ds.frames_robot(rid), frames_total=ds.frames_total,
        queries=labels, vox_count=n_vox, ray_count=n_ray, ts=ts)
      ms.publish_status(rid, status)

  def _status_loop(self):
    while not self._status_stop.wait(self.status_period_s):
      try:
        self.publish_status_once()
      except Exception:
        logger.exception("Status publish failed.")

  # ------------------------------------------------------------------ #
  # Main loop
  # ------------------------------------------------------------------ #

  @torch.inference_mode()
  def run(self):
    total_wall_t0 = time.time()
    total_map = 0
    total_frames_processed = 0
    wall_t0 = time.time()

    stream = list()
    with self._status_lock:
      if self.status == MappingServer.Status.INIT:
        self.status = MappingServer.Status.MAPPING
        # NO DataLoader: frames from different robots must not be collated,
        # and batch_size is 1 by construction here.
        stream = self.dataset
        logger.info("Datastream opened for %s. Starting shared mapping.",
                    [mrc.robot_name(r) for r in self.dataset.robot_ids])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, frame in enumerate(stream):
      if frame is None:
        break
      robot_id = int(frame.get("robot_id", self.dataset.robot_ids[0]))

      rgb_img = frame["rgb_img"].unsqueeze(0).to(device)
      pose_4x4 = frame["pose_4x4"].unsqueeze(0).to(device)
      kwargs = dict()
      if "confidence_map" in frame:
        kwargs["conf_map"] = frame["confidence_map"].unsqueeze(0).to(device)

      if "depth_img" not in frame and self.depth_estimator is None:
        raise ValueError(
          "Dataset did not return 'depth_img' and no depth_estimator is "
          "configured.")
      depth_img = (frame["depth_img"].unsqueeze(0).to(device)
                   if "depth_img" in frame else None)

      if self.depth_estimator is not None:
        depth_init = None if depth_img is None else depth_img.squeeze(1)
        refined_depth = self.depth_estimator.estimate_depth(
          rgb_image=rgb_img, depth_init=depth_init, pose_4x4=pose_4x4,
          intrinsics_3x3=self.dataset.intrinsics_3x3)
        depth_img = refined_depth.to(rgb_img.device).unsqueeze(1)

      if self.cfg.depth_limit >= 0:
        depth_img[torch.logical_and(
          torch.isfinite(depth_img),
          depth_img > self.cfg.depth_limit)] = torch.inf

      if self.vis is not None:
        if hasattr(self.vis, "set_active_robot"):
          self.vis.set_active_robot(robot_id)
        if self.cfg.vis.pose_period > 0 and i % self.cfg.vis.pose_period == 0:
          self.vis.log_pose(frame["pose_4x4"])
        if self.cfg.vis.input_period > 0 and i % self.cfg.vis.input_period == 0:
          self.vis.log_img(frame["rgb_img"].permute(1, 2, 0))
          self.vis.log_depth_img(depth_img.cpu()[-1].squeeze())

      map_t0 = time.time()
      r = self.mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
      map_t1 = time.time()

      if self.vis is not None:
        if self.cfg.vis.input_period > 0 and i % self.cfg.vis.input_period == 0:
          self.mapper.vis_update(**r)
        if self.cfg.vis.map_period > 0 and i % self.cfg.vis.map_period == 0:
          self.mapper.vis_map()

      if self.cfg.querying.period > 0 and i % self.cfg.querying.period == 0:
        self.run_queries()

      if (self.messaging_service is not None
          and self.cfg.messaging_publish_period > 0
          and i % self.cfg.messaging_publish_period == 0):
        self._publish_map_pc()

      if self.vis is not None:
        self.vis.step()

      total_frames_processed += 1
      map_p = max(map_t1 - map_t0, 1e-9)
      total_map += map_p
      wall_t1 = time.time()
      wall_p = max(wall_t1 - wall_t0, 1e-9)
      wall_t0 = wall_t1
      logger.info("[#%4d#][%s] Wall (#%6.4f# ms - #%6.2f# frame/s), "
                  "Mapping (#%6.4f# ms - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)",
                  i, mrc.robot_name(robot_id), wall_p * 1e3, 1.0 / wall_p,
                  map_p * 1e3, 1.0 / map_p, map_p / wall_p * 100)

      with self._status_lock:
        if self.status != MappingServer.Status.MAPPING:
          logger.info("Mapping stopped.")
          break

    total_wall = time.time() - total_wall_t0
    if total_map > 0 and total_wall > 0:
      logger.info("Total Wall (#%6.4f# s - #%6.2f# frame/s), "
                  "Total Mapping (#%6.4f# s - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)",
                  total_wall, total_frames_processed / total_wall,
                  total_map, total_frames_processed / total_map,
                  total_map / total_wall * 100)

    # Same idle/shutdown dance as the single-robot server.
    self._status_lock.acquire()
    if self.status == MappingServer.Status.MAPPING:
      if self.messaging_service is not None:
        self.status = MappingServer.Status.IDLE
        try:
          self.dataset.shutdown()
        except AttributeError:
          pass
      else:
        self._status_lock.release()
        self.shutdown()
        return

    if not self.cfg.querying.compute_prob:
      self._queries_feats = None
      if self._queries_labels:
        self._queries_labels.clear()
    while self.status == MappingServer.Status.IDLE:
      self._status_lock.release()
      time.sleep(1)
      with self._query_lock:
        if self._queries_updated:
          self.run_queries()
      self._status_lock.acquire()

    self.status = MappingServer.Status.CLOSED
    self._status_lock.release()
    self.shutdown()

  def shutdown(self):
    stop = getattr(self, "_status_stop", None)
    if stop is not None:
      stop.set()
    try:
      encoder = self.encoder
    except AttributeError:
      encoder = None
    super().shutdown()
    # ClientEncoder holds a socket; the real encoders do not define close().
    if encoder is not None and hasattr(encoder, "close"):
      try:
        encoder.close()
      except Exception:
        logger.exception("Failed to close the encoder connection.")


def _signal_handler(server, sig, frame):
  with server._status_lock:
    if server.status == MappingServer.Status.MAPPING:
      if server.messaging_service is not None:
        logger.info("Received interrupt. Stopping mapping; messaging service "
                    "still online. Interrupt again to shut down.")
        server.status = MappingServer.Status.IDLE
      else:
        logger.info("Received interrupt. Shutting down.")
        server.status = MappingServer.Status.CLOSING
    elif server.status == MappingServer.Status.IDLE:
      logger.info("Received interrupt. Shutting down.")
      server.status = MappingServer.Status.CLOSING
  try:
    server.dataset.shutdown()
  except AttributeError:
    pass


@hydra.main(version_base=None, config_path="configs",
            config_name="shared_humans")
@torch.inference_mode()
def main(cfg=None):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
  if cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

  # A missing background file should degrade the vocabulary, not kill the run.
  qf = cfg.querying.query_file
  if qf is not None and not os.path.exists(str(qf)):
    logger.error("querying.query_file %s does not exist. Continuing WITHOUT a "
                 "background vocabulary -- with compute_prob=True that makes "
                 "every softmax score near 1.0 and the map useless for "
                 "thresholding. Fix the path.", qf)
    cfg.querying.query_file = None

  try:
    server = MultiRobotMappingServer(cfg)
  except KeyboardInterrupt:
    logger.info("Shutdown before initializing completed.")
    return

  signal.signal(signal.SIGINT, partial(_signal_handler, server))
  try:
    server.run()
  except torch.OutOfMemoryError:
    # TEST-ONLY recovery contract (2026-09-02, user): on CUDA OOM this
    # process exits IMMEDIATELY with code 99 and offboard_compute.sh's
    # supervisor loop starts a fresh server. The in-RAM map is lost — the
    # bagged /robot_*/rayfronts/msg_serv/{voxels,rays}_sim/all snapshots are
    # the analysis record across restarts — but a best-effort dump of the
    # semantic voxel tensors is written first so the map state at the moment
    # of death is not.
    logger.critical("[oom-restart] CUDA OOM — dumping map and exiting 99 "
                    "for the supervisor to restart us.")
    try:
      torch.cuda.empty_cache()
      dump = {}
      for attr in ("global_vox_xyz", "global_vox_rgb_feat_cnt",
                   "global_rays_orig_angles", "global_rays_feats_cnt"):
        t = getattr(server.mapper, attr, None)
        if t is not None:
          dump[attr] = t.detach().cpu()
      if dump:
        path = time.strftime("/tmp/offboard/map_dump_oom_%Y%m%d_%H%M%S.pt")
        torch.save(dump, path)
        logger.critical("[oom-restart] map dumped to %s (%s)", path,
                        {k: tuple(v.shape) for k, v in dump.items()})
    except Exception:                                     # noqa: BLE001
      logger.exception("[oom-restart] map dump failed; exiting anyway")
    os._exit(99)
  except Exception:
    server.shutdown()
    raise


if __name__ == "__main__":
  def _cleanup():
    # nanobind leak cleanup, same as mapping_server.
    import typing
    for c in typing._cleanups:
      c()
  atexit.register(_cleanup)
  main()
