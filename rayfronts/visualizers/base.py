"""Defines an abstract class for 3D mapping visualization."""

import abc
import logging
from functools import wraps
from collections import defaultdict
from typing import Tuple

import torch

from rayfronts import utils, feat_compressors

logger = logging.getLogger(__name__)

class Mapping3DVisualizer(abc.ABC):
  """Base interface for all 3D mapping visualizers
  
  Attributes:
    intrinsics_3x3: See __init__
    img_size: See __init__
    base_point_size: See __init__
    global_heat_scale: See __init__
    feat_compressor: See __init__
    device: Device to use for computation.
    time_step: Current visualization step
  """

  def __init__(self, intrinsics_3x3: torch.FloatTensor,
               img_size = None,
               base_point_size: float = None,
               global_heat_scale: bool = False,
               feat_compressor: feat_compressors.FeatCompressor = None,
               device: str = None,
               **kwargs):
    """

    Args:
      intrinsics_3x3: A 3x3 float tensor including camera intrinsics.
      img_size: If not set to None, then all 2D images will be resized before
        being logged. Useful to reduce memory requirement or to ensure all 2D
        images being logged have the same size and are aligned.
      base_point_size: size of points/voxels in world units. Set to None to
        leave it up to the visualizer to set a reasonable default.
      global_heat_scale: Whether to use the same heatmap scale across all logs.
      feat_compressor: Compressor used to compress features before visualization
        If set to None, then PCA is computed on the first feature vis call.
    """
    self.intrinsics_3x3 = intrinsics_3x3
    self.img_size = img_size
    self.base_point_size = base_point_size
    self.global_heat_scale = global_heat_scale

    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    self.time_step = 0

    if feat_compressor is None:
      self.feat_compressor = feat_compressors.PcaCompressor(3)
    else:
      self.feat_compressor = feat_compressor

    # Running max and min to use for normalizing the features for visualization.
    self._projected_feats_max = None
    self._projected_feats_min = None

    # Running max and min heat to use for normalizing heatmaps consistently.
    # Only used when global_heat_scale is set to True.
    self._heat_max = None
    self._heat_min = None

  def _preprocess_img(self, img: torch.FloatTensor) -> torch.FloatTensor:
    """Resize image to self.img_size if needed
    
    Args:
      img: (HxWx3) Float tensor in the range [0-1].
    
    Returns:
      (H'xW'x3) Float tensor in the range [0-1].
    """
    src_h, src_w = img.shape[:2]
    if self.img_size is not None:
      try:
        tgt_h, tgt_w = self.img_size
      except TypeError:
        tgt_h = tgt_w = self.img_size

      if tgt_h != src_h or tgt_w != src_w:
        if len(img.shape) == 3:
          img = img.permute(2, 0, 1).unsqueeze(0)
        return torch.nn.functional.interpolate(
          img, size = (tgt_h, tgt_w),
          mode = "bilinear").squeeze().permute(1, 2, 0)
    return img

  ## Basic logging primitives

  @abc.abstractmethod
  def log_pose(self,
               pose_4x4: torch.FloatTensor,
               layer: str = "pose") -> None:
    """Pose logging primitive. Must be implemented by child class.

    Args:
      pose_4x4: (4x4) float tensor. Pose is a 4x4 float32 tensor in opencv RDF.
        a pose is the extrinsics transformation matrix that transforms a point
        from camera/robot coordinates to world coordinates. Last row should
        always be [0, 0, 0, 1].
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    pass

  @abc.abstractmethod
  def log_img(self,
              img: torch.FloatTensor,
              layer: str = "img",
              pose_layer: str = "pose") -> None:
    """Image logging primitive. Must be implemented by child class.
    
    Args:
      img: (HxWx3) Float tensor in the range [0-1] corresponding to RGB values.
      layer: Name of layer to log to. Interpretation varies by child class.
      pose_layer: Name of pose layer to associate this image with.
    """
    pass

  @abc.abstractmethod
  def log_pc(self,
             pc_xyz: torch.FloatTensor,
             pc_rgb: torch.FloatTensor = None,
             pc_radii: torch.FloatTensor = None,
             layer: str = "pc") -> None:
    """Point cloud logging primitive. Must be implemented by child class.

    Args:
      pc_xyz: (Nx3) float tensor
      pc_rgb: (Nx3) float tensor from 0-1 or (Nx4) for an optional alpha channel
      pc_radii: (Nx1) float tensor of values [0 - +inf] describing the radius of
        each point. The implementation may choose to ignore this.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    pass

  @abc.abstractmethod
  def log_arrows(self,
                 arr_origins: torch.FloatTensor,
                 arr_dirs: torch.FloatTensor,
                 arr_rgb: torch.FloatTensor = None,
                 layer: str = "arr") -> None:
    """Arrow logging primitive. Must be implemented by child class.
    
    Args:
      arr_origins: (Nx3) Float tensor represeting starting points of arrows
      arr_dirs: (Nx3) Float tensor represeting direction vectors.
        arr_origins+arr_dirs gives you the end point of the arrow.
      arr_rgb: (Nx3) (Optional) colors for arrows.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    pass

  @abc.abstractmethod
  def log_box(self,
              box_mins: torch.FloatTensor,
              box_maxs: torch.FloatTensor,
              layer: str = "box"):
    """Bounding box logging primitive. Must be implemented by child class.
    
    Args:
      box_mins: (Nx3) Float tensor describing the min value corner of the boxes.
      box_maxs: (Nx3) Float tensor describing the max value corner of the boxes.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    pass

  ## Label variants of logging.

  # TODO: Provide default implementations that do category to rgb then logs
  # using the corresponding primitive e.g log_pc.

  def log_label_img(self,
                    img_label: torch.LongTensor,
                    layer: str = "img_label",
                    pose_layer: str = "pose") -> None:
    """Logs a label image where each pixel has a numerical label.

    Maps labels to a variety of disparate RGB colors consistent with other
    label logging functions.

    Args:
      img_label: (HxW) Long tensor where each pixel corresponds to a label.
      layer: Name of layer to log to. Interpretation varies by child class.
      pose_layer: Name of pose layer to associate this image with.
    """
    raise NotImplementedError()

  def log_label_pc(self,
                   pc_xyz: torch.FloatTensor,
                   pc_labels: torch.LongTensor,
                   layer = "pc_label") -> None:
    """Logs a label point cloud where each point has a numerical label.

    Maps labels to a variety of disparate RGB colors consistent with other
    label logging functions.

    Args:
      pc_xyz: (Nx3) Float tensor
      pc_labels: (N) Long tensor describing the labels of pc_xyz.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    raise NotImplementedError()

  def log_label_arrows(self,
                       arr_origins: torch.FloatTensor,
                       arr_dirs: torch.FloatTensor,
                       arr_labels: torch.LongTensor = None,
                       layer: str = "arr_label") -> None:
    """Logs arrows where each arrow has a numerical label.
    
    Maps labels to a variety of disparate RGB colors consistent with other
    label logging functions.

    Args:
      arr_origins: (Nx3) Float tensor represeting starting points of arrows
      arr_dirs: (Nx3) Float tensor represeting direction vectors.
        arr_origins+arr_dirs gives you the end point of the arrow.
      arr_labels: (N) Long tensor describing the labels of the arrows.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    raise NotImplementedError()

  ## Heat variants of logging.

  def _normalize_heat(self, heat: torch.FloatTensor) -> torch.FloatTensor:
    """Normalizes a list of heat values to be within [0-1] range.
    
    If global_heat_scale is True, then keeps a running min and max for
    normalizing all subsequent calls with the same scale.

    Args:
      heat: An arbitrary shaped Float Tensor.
    Returns:
      The same heat tensor but normalized to be within [0-1] range.
    """
    if not self.global_heat_scale:
      return utils.norm_01(heat)

    mn = heat.min()
    mx = heat.max()
    if self._heat_max is None:
      self._heat_max = mx
      self._heat_min = mn
    else:
      self._heat_max = torch.maximum(mx, self._heat_max)
      self._heat_min = torch.minimum(mn, self._heat_min)

    heat = heat - self._heat_min
    scale = self._heat_max - self._heat_min
    heat = heat / scale
    return heat

  def log_heat_img(self,
                   img_heat: torch.LongTensor,
                   high_color: Tuple[int] = (253, 231, 37),
                   low_color: Tuple[int] = (68, 1, 84),
                   layer: str = "img_heat",
                   pose_layer: str = "pose") -> None:
    """Logs a heat map.
    
    Args:
      img_heat: (HxW) Float Tensor with heat values. Need not be normalized.
      high_color: A tuple of 3 integers describing the RGB values of the highest
        value in the heatmap.
      low_color: A tuple of 3 integers describing the RGB values of the lowest
        value in the heatmap.
      layer: Name of layer to log to. Interpretation varies by child class.
      pose_layer: Name of pose layer to associate this image with.
    """
    raise NotImplementedError()

  def log_heat_pc(self,
                  pc_xyz: torch.FloatTensor,
                  pc_heat: torch.FloatTensor = None,
                  high_color: Tuple[int] = (253, 231, 37),
                  low_color: Tuple[int] = (68, 1, 84),
                  vis_thresh: float = 0,
                  scale_size: bool = False,
                  layer: str = "pc_heat") -> None:
    """Logs a heat point cloud.
    
    Args:
      pc_xyz: (Nx3) float tensor
      pc_heat: (N) float tensor describing the heat values.
      high_color: A tuple of 3 integers describing the RGB values of the highest
        value in the heatmap.
      low_color: A tuple of 3 integers describing the RGB values of the lowest
        value in the heatmap.
      vis_thresh: If [0-1] normalized heat is less than this threshold, the
        corresponding points will not be logged.
      scale_size: Whether the point size should scale with the normalized heat.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    if pc_xyz.shape[0] == 0:
      return
    pc_heat = self._normalize_heat(pc_heat).unsqueeze(-1)
    c1 = torch.tensor(high_color, dtype=torch.float, device=pc_heat.device)
    c1 = c1.unsqueeze(0)/255
    c2 = torch.tensor(low_color, dtype=torch.float, device=pc_heat.device)
    c2 = c2.unsqueeze(0)/255
    cmapped_pc = pc_heat * c1 + (1-pc_heat) * c2
    radii = self.base_point_size
    if scale_size:
      radii = radii * pc_heat.squeeze(-1)
    if vis_thresh > 0:
      pc_xyz = pc_xyz[pc_heat.squeeze(-1) > vis_thresh, :]
      cmapped_pc = cmapped_pc[pc_heat.squeeze(-1) > vis_thresh, :]
    self.log_pc(pc_xyz, cmapped_pc, radii, layer=layer)

  def log_heat_arrows(self,
                      arr_origins: torch.FloatTensor,
                      arr_dirs: torch.FloatTensor,
                      arr_heat: torch.FloatTensor,
                      high_color: Tuple[int] = (253, 231, 37),
                      low_color: Tuple[int] = (68, 1, 84),
                      vis_thresh: float = 0,
                      scale_size: bool = False,
                      layer: str = "arr_heat") -> None:
    """Logs heat arrows.
    
    Args:
      arr_origins: (Nx3) Float tensor represeting starting points of arrows
      arr_dirs: (Nx3) Float tensor represeting direction vectors.
        arr_origins+arr_dirs gives you the end point of the arrow.
      arr_heat: (N) Float tensor describing the heat values of the arrows.
      high_color: A tuple of 3 integers describing the RGB values of the highest
        value in the heatmap.
      low_color: A tuple of 3 integers describing the RGB values of the lowest
        value in the heatmap.
      vis_thresh: If [0-1] normalized heat is less than this threshold, the
        corresponding arrows will not be logged.
      scale_size: Whether the arrow size should scale with the normalized heat.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    if arr_origins.shape[0] == 0:
      return
    arr_heat = self._normalize_heat(arr_heat).unsqueeze(-1)
    c1 = torch.tensor(high_color, dtype=torch.float, device=arr_heat.device)
    c1 = c1.unsqueeze(0)/255
    c2 = torch.tensor(low_color, dtype=torch.float, device=arr_heat.device)
    c2 = c2.unsqueeze(0)/255
    cmapped = arr_heat * c1 + (1-arr_heat) * c2
    if scale_size:
      arr_dirs = arr_dirs*torch.clip(arr_heat, min=0.3)
    if vis_thresh > 0:
      arr_origins = arr_origins[arr_heat.squeeze(-1) > vis_thresh, :]
      arr_dirs = arr_dirs[arr_heat.squeeze(-1) > vis_thresh, :]
      cmapped = cmapped[arr_heat.squeeze(-1) > vis_thresh, :]
    self.log_arrows(arr_origins, arr_dirs, cmapped, layer=layer)

  ## Featurized variants of logging

  def _normalize_projected_feats(
      self, proj_feats: torch.FloatTensor) -> torch.FloatTensor:
    """Normalizes features already projected to 3 values.

    Keeps track of normalizing values to use across calls.

    Args:
      proj_feats (*x3) Float tensor.
    Returns:
      Float tensor with RGB values in [0-1] range.
    """
    assert proj_feats.shape[-1] == 3
    projected_feats_min = torch.min(
      proj_feats.reshape(-1, 3), dim=0).values
    projected_feats_max = torch.max(
      proj_feats.reshape(-1, 3), dim=0).values

    if self._projected_feats_min is None:
      self._projected_feats_max = projected_feats_max
      self._projected_feats_min = projected_feats_min
    else:
      self._projected_feats_min = torch.minimum(projected_feats_min,
                                               self._projected_feats_min)
      self._projected_feats_max = torch.maximum(projected_feats_max,
                                               self._projected_feats_max)

    proj_feats = proj_feats - self._projected_feats_min
    scale = self._projected_feats_max - self._projected_feats_min
    proj_feats = proj_feats / scale
    return proj_feats

  def _project_feats_to_rgb(
      self, feats: torch.FloatTensor) -> torch.FloatTensor:
    """Projects features to RGB space for visualization.

    Uses the provided feature compressor or PCA to get 3 channels corresponding
    to RGB. If the feature compressor is not fitted, will attempt to fit it
    with 'feats'. If it fails, then it just selects the first 3 features. If
    it succeeds, it uses the feature compressor to compress the features then
    select the first 3 features for visualization.

    Args:
      feats: (*xC) Float tensor describing features.
    """
    C = feats.shape[-1]
    assert C > 0
    output_shape = list(feats.shape)
    output_shape[-1] = 3
    if C > 3:
      if self.feat_compressor.is_fitted():
        feats = self.feat_compressor.compress(feats)[..., :3]
      else:
        try:
          self.feat_compressor.fit(feats)
          # Fitted compressor to new distribution. Let's reset the min max 
          # for normalization.
          self._projected_feats_min = None
          self._projected_feats_max = None
          feats = self.feat_compressor.compress(feats)[..., :3]
        except:
          feats = feats[..., :3]
    elif C < 3:
      feats_flat = torch.nn.functional.pad(feats_flat, (0, 0, 0, 3-C))
      feats = feats_flat.reshape(*output_shape)

    return self._normalize_projected_feats(feats)

  def log_feature_img(self,
                      feat_img: torch.FloatTensor,
                      layer: str = "img_feat",
                      pose_layer: str = "pose") -> None:
    """Logs a feature image.

    Args:
      feat_img: (HxWxC) Float tensor describing the feature image.
      layer: Name of layer to log to. Interpretation varies by child class.
      pose_layer: Name of pose layer to associate this image with.
    """
    feat_img = self._project_feats_to_rgb(feat_img)
    self.log_img(self._preprocess_img(feat_img),
                       layer=layer, pose_layer=pose_layer)

  def log_feature_pc(self,
                     pc_xyz: torch.FloatTensor,
                     pc_feat: torch.FloatTensor,
                     layer: str = "pc_feat") -> None:
    """Logs a point cloud colorized by arbitrary features.

    Args:
      pc_xyz: (Nx3) float tensor
      pc_feat: (NxC) float tensor describing the feature values.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    pc_feat = self._project_feats_to_rgb(pc_feat)
    self.log_pc(pc_xyz, pc_feat, layer=layer)

  def log_feature_arr(self,
                      arr_origins: torch.FloatTensor,
                      arr_dirs: torch.FloatTensor,
                      arr_feats: torch.FloatTensor = None,
                      layer: str = "arr_feat") -> None:
    """Logs arrows colorized by arbitrary features.

    Args:
      arr_origins: (Nx3) Float tensor represeting starting points of arrows
      arr_dirs: (Nx3) Float tensor represeting direction vectors.
        arr_origins+arr_dirs gives you the end point of the arrow.
      arr_feats: (NxC) Float tensor describing the features of the arrows.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    arr_feats = self._project_feats_to_rgb(arr_feats)
    self.log_arrows(arr_origins, arr_dirs, arr_feats, layer=layer)

  ## Special variants

  def log_occ_pc(self,
                 pc_xyz: torch.FloatTensor,
                 pc_occ: torch.FloatTensor,
                 pc_radii: torch.FloatTensor = None,
                 layer: str = "pc_occ"):
    """Log a point cloud / sparse voxel map representing occupancy.
    
    Args:
      pc_xyz: (Nx3) Float tensor
      pc_occ: N Float tensor representing occupancy values.
        Need not be normalized.
      pc_radii: (Optional) N Float tensor describing radius of each point.
      layer: Name of layer to log to. Interpretation varies by child class.
    """
    color = torch.tensor([0.8, 0.2, 0.8],
                         dtype=torch.float, device=pc_occ.device)
    colors = torch.empty((pc_xyz.shape[0], 4),
                         dtype=torch.float, device=pc_occ.device)
    colors[:, :3] = color
    if pc_occ.shape[0] > 0:
      colors[:, -1:] = utils.norm_01(pc_occ)
    if pc_radii is None:
      pc_radii = (colors[:, -1]+0.2)/1.2*self.base_point_size
    self.log_pc(pc_xyz, colors, pc_radii, layer)

  def log_depth_img(self,
                    depth_img: torch.FloatTensor,
                    layer: str = "img_depth",
                    pose_layer: str = "pose"):
    """Logs depth images.

    Non-binding default implementation does the following mappings:
    +inf mapped to red
    -inf mapped to green
    nan mapped to blue
    Finite values mapped to grayscale.
    Child classes may choose to ignore this mapping.

    Args:
      depth_img: HxW float tensor describing depth in world scale.
      layer: Name of layer to log to. Interpretation varies by child class.
      pose_layer: Name of pose layer to associate this image with.
    """
    self.img_updated = True
    H,W = depth_img.shape
    rgb_depth_img = torch.zeros(size=(H, W, 3), device=depth_img.device,
                                dtype=torch.float)
    mask = depth_img.isfinite()
    # Frames can be entirely non-finite (e.g. depth_limit clamps everything
    # beyond range to inf at altitude); norm_01 on an empty tensor throws.
    if mask.any():
      rgb_depth_img[mask, :] = utils.norm_01(depth_img[mask]).unsqueeze(-1)
    mask = depth_img.isposinf()
    rgb_depth_img[mask, :] = torch.tensor([1., 0, 0])
    mask = depth_img.isneginf()
    rgb_depth_img[mask, :] = torch.tensor([0, 1., 0])
    mask = depth_img.isnan()
    rgb_depth_img[mask, :] = torch.tensor([0, 0, 1.])
    self.log_img(rgb_depth_img, layer=layer, pose_layer=pose_layer)

  ## Other

  def step(self):
    """Advance visualizer time by one unit."""
    self.time_step += 1
