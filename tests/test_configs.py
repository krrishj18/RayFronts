"""Hydra composition of the shared-server configs.

Composition only -- nothing is instantiated, so no model is loaded and no ROS
node is created. Runs on the host with
``uv run --with hydra-core python -m pytest tests/test_configs.py``.
"""

import os
import pathlib

import pytest

hydra = pytest.importorskip("hydra", reason="needs hydra-core")
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from conftest import CONFIGS, REPO_CONFIGS  # noqa: E402

PKG_CONFIGS = str(CONFIGS)
SIDE_CONFIGS = str(REPO_CONFIGS)


def _compose(config_name, overrides=(), primary_dir=SIDE_CONFIGS,
             extra_dir=PKG_CONFIGS):
  """Mirror `--config-dir <side> --config-name <name>` on the real CLI."""
  ov = [f"hydra.searchpath=[file://{extra_dir}]"] + list(overrides)
  with initialize_config_dir(version_base=None, config_dir=primary_dir):
    return compose(config_name=config_name, overrides=ov,
                   return_hydra_config=False)


# --------------------------------------------------------------------------- #
# The new shared config
# --------------------------------------------------------------------------- #

def test_shared_humans_composes():
  cfg = _compose("shared_humans")
  assert cfg.dataset._target_ == \
      "rayfronts.datasets.MultiRobotRos2Subscriber"
  assert cfg.messaging_service._target_ == \
      "rayfronts.messaging_services.MultiRobotRos2MessagingService"
  assert cfg.vis._target_ == "rayfronts.visualizers.MultiRobotRos2Vis"
  assert cfg.encoder._target_ == "rayfronts.image_encoders.ClientEncoder"
  assert cfg.mapping._target_ == "rayfronts.mapping.SemanticRayFrontiersMap"


def test_shared_humans_cli_overrides_from_the_contract():
  """Exactly the command line the launcher runs."""
  cfg = _compose("shared_humans", [
    "dataset.robot_ids=[1,2]",
    "encoder=client",
    "encoder.socket=/tmp/rayfronts/encoder.sock",
  ])
  assert list(cfg.dataset.robot_ids) == [1, 2]
  assert cfg.encoder.socket == "/tmp/rayfronts/encoder.sock"
  # The messaging service and visualizer must follow the dataset without
  # having to be overridden separately.
  assert list(cfg.messaging_service.robot_ids) == [1, 2]
  assert list(cfg.vis.robot_ids) == [1, 2]
  assert cfg.messaging_service.domain_ids is None
  assert cfg.vis.domain_ids is None


def test_domain_ids_propagate_when_overridden():
  cfg = _compose("shared_humans",
                 ["dataset.robot_ids=[1,2]", "dataset.domain_ids=[11,12]"])
  assert list(cfg.messaging_service.domain_ids) == [11, 12]
  assert list(cfg.vis.domain_ids) == [11, 12]


def test_frozen_topic_templates():
  cfg = _compose("shared_humans")
  ms = cfg.messaging_service
  assert ms.topic_prefix_template == "/{robot}/rayfronts/msg_serv"
  assert ms.text_query_topic_template == \
      "/{robot}/rayfronts/msg_serv/new_text_query"
  assert ms.guiding_queries_topic_template == \
      "/{robot}/rayfronts/msg_serv/guiding_queries"
  assert ms.status_topic_template == "/{robot}/rayfronts/status"
  assert cfg.vis.topic_prefix_template == "/{robot}/rayfronts"


def test_dataset_topic_templates_match_the_isaac_zed_topics():
  cfg = _compose("shared_humans")
  d = cfg.dataset
  assert d.rgb_topic == "/{robot}/sensors/front_stereo/left/image_rect"
  assert d.depth_topic == \
      "/{robot}/sensors/front_stereo/left/depth_ground_truth"
  assert d.intrinsics_topic == "/{robot}/sensors/front_stereo/left/camera_info"
  assert d.pose_topic == "/{robot}/odometry_conversion/odometry"
  assert d.pose_msg_type == "odometry"
  assert d.navsat_topic == "/{robot}/interface/mavros/global_position/global"
  assert d.anchor_mode == "gps"
  assert d.src_coord_system == "flu"


def test_small_object_tuning_differs_from_low_memory():
  shared = _compose("shared_humans")
  low = _compose("low_memory")
  assert low.mapping.max_pts_per_frame == 1000
  assert shared.mapping.max_pts_per_frame == 1500
  assert low.mapping.sem_pruning_thresh == 5
  assert shared.mapping.sem_pruning_thresh == 1
  assert low.mapping.sem_pruning_period == 32
  assert shared.mapping.sem_pruning_period == 8
  # DIVERGED 2026-09-02 night (user): vox 0.3 for real body shape, caps cut
  # to pay for the ~4.6x voxel-count increase.
  assert low.mapping.vox_size == 0.50
  assert shared.mapping.vox_size == 0.30
  assert low.mapping.max_rays_per_frame == 500
  assert shared.mapping.max_rays_per_frame == 200
  assert low.mapping.max_empty_pts_per_frame == 3000
  assert shared.mapping.max_empty_pts_per_frame == 2000
  assert shared.mapping.fronti_subsampling == low.mapping.fronti_subsampling
  assert (shared.mapping.fronti_subsampling_min_fronti
          == low.mapping.fronti_subsampling_min_fronti)
  # DIVERGED 2026-09-02: low_memory's [480,480] square-stretches the 960x600
  # camera 1.6x vertically; shared runs aspect-correct 960x608. (A 640x400
  # pass was tried and REVERTED the same day: with PCA the background
  # cosines collapsed and the softmax painted the whole map as person —
  # PCA solves the VRAM problem, resolution buys the semantics.)
  assert list(low.dataset.rgb_resolution) == [480, 480]
  assert list(shared.dataset.rgb_resolution) == [608, 960]
  assert list(shared.dataset.depth_resolution) == [608, 960]
  assert shared.dataset.frame_skip == low.dataset.frame_skip == 10


def test_querying_is_usable_out_of_the_box():
  cfg = _compose("shared_humans")
  # low_memory leaves this null and relies on a CLI override; a shared server
  # started without it would raise on the first query.
  assert cfg.querying.text_query_mode == "prompts"
  assert cfg.querying.compute_prob is True
  # False is REQUIRED with the PCA compressor: its basis lives in the
  # 768-d pre-alignment image space (fit at mapping/base.py:310) while text
  # queries are 1152-d language vectors — feature_query bridges the spaces
  # by decompressing then aligning, which only the compressed=False path
  # does (live crash loop 2026-09-02, "(6x1152 and 768x100)").
  assert cfg.querying.compressed is False
  # 256, not low_memory's 100: the single-frame online fit at 100 dims
  # reconstructs everything toward the frame mean (person-everywhere
  # softmax, live 2026-09-02).
  assert cfg.feat_compressor.out_dim == 192
  assert cfg.batch_size == 1
  assert cfg.status_period_s == 1.0


def test_query_file_seeds_the_vocabulary_person_first():
  """query_file is BACK since 2026-09-02 evening, on purpose: topic-delivered
  vocabulary proved untrustworthy (restarted mapper incarnations'
  new_text_query subscriber repeatedly came up deaf), while file seeding at
  startup needs no delivery and — with the ordered dedupe in add_queries —
  pins the column order deterministically. The file MUST lead with the
  mission's single target: person = sim_0 (query "person" — the OG single
  query; a person+casualty two-positive pass was tried 2026-09-02 and
  REVERTED: with debris in the background casualty adds nothing at buried
  bodies, and the summed gate ran looser than its single-positive
  calibration, spraying ray-tier FP boxes). Everything after person is the
  background set, debris included (the swept trade-off: debris suppresses
  clutter FPs at the cost of buried casualties — accepted)."""
  cfg = _compose("shared_humans")
  assert str(cfg.querying.query_file).endswith("vocab_person_first.txt")
  local = REPO_CONFIGS / "vocab_person_first.txt"
  assert local.exists()
  labels = [l.strip() for l in local.read_text().splitlines() if l.strip()]
  # The FINAL vocabulary (user 2026-09-02): person + road, grass, tree,
  # house, debris, sky.
  assert labels == ["person", "road", "grass", "tree", "house", "wall",
                    "debris", "sky", "car", "utility pole",
                    "street light", "hydrant", "crosswalk"]


# --------------------------------------------------------------------------- #
# The encoder server
# --------------------------------------------------------------------------- #

def test_encoder_server_config_composes():
  cfg = _compose("encoder_server", primary_dir=PKG_CONFIGS,
                 extra_dir=SIDE_CONFIGS)
  assert cfg.encoder._target_ == "rayfronts.image_encoders.RADSegEncoder"
  assert cfg.encoder_server.socket == "/tmp/rayfronts/encoder.sock"
  assert cfg.encoder_server.device is None
  assert cfg.encoder_server.warmup is None


def test_encoder_server_cli_overrides_from_the_contract():
  cfg = _compose("encoder_server",
                 ["encoder=radseg",
                  "encoder_server.socket=/tmp/rayfronts/encoder.sock"],
                 primary_dir=PKG_CONFIGS, extra_dir=SIDE_CONFIGS)
  assert cfg.encoder_server.socket == "/tmp/rayfronts/encoder.sock"
  assert cfg.encoder._target_ == "rayfronts.image_encoders.RADSegEncoder"


def test_encoder_server_serves_the_dummy_encoder_too():
  cfg = _compose("encoder_server", ["encoder=dummy"],
                 primary_dir=PKG_CONFIGS, extra_dir=SIDE_CONFIGS)
  assert cfg.encoder._target_ == "rayfronts.image_encoders.DummyEncoder"
  assert cfg.encoder.patch_size == 16


def test_client_encoder_defaults():
  cfg = _compose("shared_humans")
  assert cfg.encoder.transport == "auto"
  assert cfg.encoder.socket == "/tmp/rayfronts/encoder.sock"
  assert float(cfg.encoder.timeout_s) > 0


# --------------------------------------------------------------------------- #
# Nothing regressed for the single-robot deployment
# --------------------------------------------------------------------------- #

def test_low_memory_still_composes_with_the_launch_file_overrides():
  """The exact overrides rayfronts.launch.xml passes today."""
  cfg = _compose("low_memory", [
    "dataset=ros2isaacsim",
    "dataset.rgb_topic=/robot_1/sensors/front_stereo/left/image_rect",
    "dataset.pose_topic=/robot_1/odometry_conversion/pose_stamped",
    "messaging_service=ros",
    "messaging_service.topic_prefix=/robot_1/rayfronts/msg_serv",
    "vis.topic_prefix=/robot_1/rayfronts",
    "querying.text_query_mode=prompts",
    "querying.query_file=null",
    "querying.compute_prob=true",
    "querying.period=10",
  ])
  assert cfg.dataset._target_ == "rayfronts.datasets.Ros2Subscriber"
  assert cfg.messaging_service._target_ == \
      "rayfronts.messaging_services.Ros2MessagingService"
  assert cfg.vis._target_ == "rayfronts.visualizers.Ros2Vis"
  assert cfg.encoder._target_ == "rayfronts.image_encoders.RADSegEncoder"


def test_background_lists_are_pure_labels():
  """MappingServer reads the file with a bare readlines(); no comments.

  A '#' line or a blank line would become a query column named '#...' or ''.
  """
  for name in ("background_humans.txt", "background_humans_min.txt"):
    path = REPO_CONFIGS / name
    labels = [l.strip() for l in path.read_text().splitlines()]
    assert labels, f"{name} is empty"
    assert all(labels), f"{name} has a blank line"
    assert not any(l.startswith("#") for l in labels), \
        f"{name} has a comment line, which would become a query column"
    assert len(labels) == len(set(labels)), f"{name} has duplicates"
    assert "person" not in labels, \
        f"{name} must not contain the target class"


def test_background_list_is_the_documented_size():
  labels = [l.strip() for l in
            (REPO_CONFIGS / "background_humans.txt").read_text().splitlines()]
  assert 25 <= len(labels) <= 40
  mini = [l.strip() for l in
          (REPO_CONFIGS / "background_humans_min.txt").read_text().splitlines()]
  assert len(mini) <= 8


# --------------------------------------------------------------------------- #
# The _target_ strings above must name classes that actually exist.
# --------------------------------------------------------------------------- #

@pytest.mark.torch
def test_every_target_resolves_to_a_real_class():
  """A typo in a `_target_` only shows up when the server is launched."""
  from conftest import HAVE_TORCH, ensure_pythonpath
  if not HAVE_TORCH:
    pytest.skip("needs torch to import the rayfronts package")
  ensure_pythonpath()
  from hydra.utils import get_class

  cfg = _compose("shared_humans")
  for group in ("dataset", "messaging_service", "vis", "encoder", "mapping",
                "feat_compressor"):
    node = cfg.get(group)
    if node is None or "_target_" not in node:
      continue
    cls = get_class(node._target_)
    assert cls is not None, group

  srv = _compose("encoder_server", primary_dir=PKG_CONFIGS,
                 extra_dir=SIDE_CONFIGS, overrides=["encoder=dummy"])
  assert get_class(srv.encoder._target_) is not None


@pytest.mark.torch
def test_the_multi_classes_satisfy_the_interfaces_the_server_expects():
  from conftest import HAVE_TORCH, ensure_pythonpath
  if not HAVE_TORCH:
    pytest.skip("needs torch to import the rayfronts package")
  ensure_pythonpath()
  from rayfronts.datasets import PosedRgbdDataset, MultiRobotRos2Subscriber
  from rayfronts.messaging_services import (MessagingService,
                                            MultiRobotRos2MessagingService)
  from rayfronts.visualizers import Mapping3DVisualizer, MultiRobotRos2Vis
  from rayfronts.image_encoders import (LangSpatialGlobalImageEncoder,
                                        ClientEncoder)
  assert issubclass(MultiRobotRos2Subscriber, PosedRgbdDataset)
  assert issubclass(MultiRobotRos2MessagingService, MessagingService)
  assert issubclass(MultiRobotRos2Vis, Mapping3DVisualizer)
  assert issubclass(ClientEncoder, LangSpatialGlobalImageEncoder)
  # No abstract methods left over -- these must be instantiable.
  for cls in (MultiRobotRos2Subscriber, MultiRobotRos2MessagingService,
              MultiRobotRos2Vis, ClientEncoder):
    assert not getattr(cls, "__abstractmethods__", frozenset()), \
        f"{cls.__name__} still has abstract methods: {cls.__abstractmethods__}"
