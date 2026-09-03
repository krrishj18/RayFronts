"""A tiny, deterministic, download-free encoder.

Exists so the encoder server / ClientEncoder transport can be exercised end to
end (including on a machine with no GPU and no network) without pulling a RADIO
checkpoint off torch.hub. It is a real ``LangSpatialGlobalImageEncoder``: every
method the mappers and the mapping server call is implemented, all outputs are
a deterministic function of the inputs, so a client/server roundtrip can be
compared for exact equality against a local instance.

Not useful for mapping anything real.
"""

from typing import List, Tuple

from typing_extensions import override
import torch

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder


class DummyEncoder(LangSpatialGlobalImageEncoder):
  """Deterministic fixed-weight encoder used for tests and dry runs.

  Attributes:
    device: See base.
    feat_dim: Channel count of the spatial/global feature space.
    lang_dim: Channel count of the language-aligned space.
    patch_size: Spatial downsampling factor of the feature map.
  """

  def __init__(self,
               device: str = None,
               feat_dim: int = 16,
               lang_dim: int = 8,
               patch_size: int = 16,
               seed: int = 17,
               compile: bool = False,
               amp: bool = False):
    super().__init__(device)
    self.feat_dim = int(feat_dim)
    self.lang_dim = int(lang_dim)
    self.patch_size = int(patch_size)
    self.seed = int(seed)
    self.compile = compile
    self.amp = amp

    g = torch.Generator(device="cpu").manual_seed(self.seed)
    # Fixed conv: 3 -> feat_dim, stride patch_size. No bias, no training.
    self._w = (torch.randn(self.feat_dim, 3, self.patch_size, self.patch_size,
                           generator=g) / (3 * self.patch_size ** 2) ** 0.5
               ).to(self.device)
    # Fixed feat_dim -> lang_dim projection.
    self._proj = (torch.randn(self.lang_dim, self.feat_dim, generator=g)
                  / self.feat_dim ** 0.5).to(self.device)

  # -- ImageEncoder --------------------------------------------------------- #

  @override
  def is_compatible_size(self, h: int, w: int) -> bool:
    return h % self.patch_size == 0 and w % self.patch_size == 0

  @override
  def get_nearest_size(self, h, w) -> Tuple[int, int]:
    p = self.patch_size
    return (max(p, int(round(h / p)) * p), max(p, int(round(w / p)) * p))

  # -- spatial / global ----------------------------------------------------- #

  @override
  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor
                               ) -> torch.FloatTensor:
    x = rgb_image.to(self.device, dtype=torch.float32)
    return torch.nn.functional.conv2d(x, self._w, stride=self.patch_size)

  @override
  def encode_image_to_vector(self, rgb_image: torch.FloatTensor
                             ) -> torch.FloatTensor:
    return self.encode_image_to_feat_map(rgb_image).mean(dim=(-1, -2))

  @override
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor):
    f = self.encode_image_to_feat_map(rgb_image)
    return f, f.mean(dim=(-1, -2))

  # -- language ------------------------------------------------------------- #

  def _text_feat(self, texts: List[str], salt: int) -> torch.FloatTensor:
    out = torch.empty((len(texts), self.lang_dim), dtype=torch.float32)
    for i, t in enumerate(texts):
      # Stable across processes: python's str hash is salted per process, so
      # build the seed out of the code points instead.
      h = salt
      for ch in str(t):
        h = (h * 131 + ord(ch)) % (2 ** 31 - 1)
      g = torch.Generator(device="cpu").manual_seed(h)
      out[i] = torch.randn(self.lang_dim, generator=g)
    out = torch.nn.functional.normalize(out, dim=-1)
    return out.to(self.device)

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    return self._text_feat(list(labels), salt=1)

  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    return self._text_feat(list(prompts), salt=2)

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor
                                           ) -> torch.FloatTensor:
    f = features.to(self.device, dtype=torch.float32)
    return torch.einsum("dc,bchw->bdhw", self._proj, f)

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor
                                          ) -> torch.FloatTensor:
    f = features.to(self.device, dtype=torch.float32)
    return f @ self._proj.T
