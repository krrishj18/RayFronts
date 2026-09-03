"""Open-set PROBE inputs: env parsing, frame conversion, nearest-voxel pick.

Everything here is the torch-free half of the probe feature — the half that
decides WHICH voxel gets logged for a ground-truth casualty position. It is
pure numpy on purpose: the values come from a mission yaml through a container
env var, so the parsers have to be total (garbage in, empty list out, never an
exception that stops a mapping server booting), and the FLU->RDF conversion is
derived from ``coord_system_transform`` rather than written down, because a
sign error there silently probes a point 25 m away and reads as "RayFronts
scores nothing at the casualty".

The torch half (probe feats never entering the query set, and the server's
torch nearest-voxel search agreeing with ``nearest_index``) is in
``test_multi_mapping_server.py``.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# RAYFRONTS_PROBE_PROMPTS
# --------------------------------------------------------------------------- #

def test_prompts_comma_separated_is_the_documented_form(mrc):
  assert mrc.parse_probe_prompts(
    "person lying down,casualty,human body") == [
      "person lying down", "casualty", "human body"]


def test_prompts_strip_whitespace_and_blanks(mrc):
  assert mrc.parse_probe_prompts("  a , ,b ,, c  ") == ["a", "b", "c"]


def test_prompts_dedupe_but_keep_first_position(mrc):
  # Position matters: the log prints the probe scores in this order.
  assert mrc.parse_probe_prompts("casualty,person,casualty") == [
    "casualty", "person"]


def test_prompts_accept_a_json_list(mrc):
  assert mrc.parse_probe_prompts('["a b", "c"]') == ["a b", "c"]


def test_prompts_accept_an_already_parsed_sequence(mrc):
  """A hydra ListConfig / a plain list from `probe_prompts:` in the yaml."""
  assert mrc.parse_probe_prompts(["a", " b ", ""]) == ["a", "b"]
  assert mrc.parse_probe_prompts(("a",)) == ["a"]


def test_prompts_of_nothing(mrc):
  assert mrc.parse_probe_prompts(None) == []
  assert mrc.parse_probe_prompts("") == []
  assert mrc.parse_probe_prompts("   ") == []
  assert mrc.parse_probe_prompts([]) == []


def test_prompts_broken_json_falls_back_to_comma_split(mrc):
  # A truncated list must not raise; the commas are still usable.
  assert mrc.parse_probe_prompts('["a", "b"') == ['["a"', '"b"']


def test_prompts_accept_bytes(mrc):
  assert mrc.parse_probe_prompts(b"a,b") == ["a", "b"]


def test_the_contrast_set_parses_to_the_same_strings_the_mission_sends(mrc):
  """RAYFRONTS_PROBE_BACKGROUND is matched against the LIVE label strings.

  semantic_search_task builds its list with
  ``[bq.strip() for bq in goal.background_queries.split(',')]``; the solo
  softmax finds the background rows by string equality against those labels,
  so the parser has to strip the space after each comma the same way. A single
  stray space here silently empties the contrast set and every solo score
  becomes 1.0.
  """
  assert mrc.parse_probe_prompts(
    "road, grass, tree, house, wood debris, sky") == [
      "road", "grass", "tree", "house", "wood debris", "sky"]


# --------------------------------------------------------------------------- #
# RAYFRONTS_PROBE_POINTS
# --------------------------------------------------------------------------- #

MISSION_POINTS = '[[-12.87,-33.52],[-15.06,-35.4],[-4.79,-69.42]]'


def test_points_the_exact_string_the_mission_yaml_ships(mrc):
  assert mrc.parse_probe_points(MISSION_POINTS) == [
    (-12.87, -33.52), (-15.06, -35.4), (-4.79, -69.42)]


def test_points_keep_their_arity(mrc):
  """2 vs 3 elements is not cosmetic: it picks horizontal vs 3D matching."""
  pts = mrc.parse_probe_points('[[1,2],[3,4,5]]')
  assert pts == [(1.0, 2.0), (3.0, 4.0, 5.0)]
  assert mrc.probe_axes(pts[0]) == mrc.RDF_HORIZONTAL_AXES
  assert mrc.probe_axes(pts[1]) == mrc.RDF_AXES


def test_points_a_fourth_number_is_dropped_not_fatal(mrc):
  assert mrc.parse_probe_points('[[1,2,3,4]]') == [(1.0, 2.0, 3.0)]


def test_points_a_single_bare_point_is_wrapped(mrc):
  assert mrc.parse_probe_points('[1.5, -2.5]') == [(1.5, -2.5)]


def test_points_accept_a_hand_typed_semicolon_string(mrc):
  assert mrc.parse_probe_points("1,2; 3,4") == [(1.0, 2.0), (3.0, 4.0)]


def test_points_accept_dicts(mrc):
  assert mrc.parse_probe_points([{"x": 1, "y": 2},
                                 {"x": 3, "y": 4, "z": 5}]) == [
    (1.0, 2.0), (3.0, 4.0, 5.0)]


def test_points_accept_an_already_parsed_sequence(mrc):
  assert mrc.parse_probe_points([[1, 2], (3, 4, 5)]) == [
    (1.0, 2.0), (3.0, 4.0, 5.0)]


@pytest.mark.parametrize("garbage", [
  None, "", "   ", "not json at all", "{", "[[",
  '["a","b"]',            # strings that are not numbers
  '[[1]]',                # one coordinate is not a point
  '[[1,"x"]]',            # a non-number in a pair
  '[null]', '[{}]', '[[]]',
  '[[1, 2, NaN]]',        # non-finite: json parses it, we must not keep it
  123, True,
])
def test_points_garbage_never_raises_and_never_invents_a_point(mrc, garbage):
  out = mrc.parse_probe_points(garbage)
  assert isinstance(out, list)
  assert out == []


def test_points_a_bad_entry_does_not_take_the_good_ones_with_it(mrc):
  assert mrc.parse_probe_points('[[1,2],"junk",[3,4]]') == [
    (1.0, 2.0), (3.0, 4.0)]


# --------------------------------------------------------------------------- #
# World FLU -> map RDF
# --------------------------------------------------------------------------- #

def test_world_flu_to_rdf_is_the_transform_not_a_hand_written_table(mrc):
  p = [7.0, -3.0, 2.0]
  np.testing.assert_allclose(
    mrc.world_flu_to_rdf(p),
    mrc.coord_system_transform("flu", "rdf") @ np.asarray(p))


def test_world_flu_to_rdf_signs(mrc):
  # rdf = (-y, -z, x): this is the inverse of the [raw top] print line
  # `px, py, pz = xyz[2], -xyz[0], -xyz[1]`.
  np.testing.assert_allclose(mrc.world_flu_to_rdf([7.0, -3.0, 2.0]),
                             [3.0, -2.0, 7.0])


def test_world_flu_to_rdf_pads_a_2d_point_with_z_zero(mrc):
  np.testing.assert_allclose(mrc.world_flu_to_rdf([-12.87, -33.52]),
                             [33.52, 0.0, -12.87])


def test_flu_rdf_round_trip(mrc):
  for p in ([1.0, 2.0, 3.0], [-12.87, -33.52, 0.0], [0.0, 0.0, 0.0]):
    np.testing.assert_allclose(
      mrc.rdf_to_world_flu(mrc.world_flu_to_rdf(p)), p, atol=1e-12)


def test_the_logger_print_line_matches_rdf_to_world_flu(mrc):
  """The `[raw top]` / `[probe@..]` lines print (z, -x, -y) inline."""
  rdf = np.array([3.0, -2.0, 7.0])
  inline = (float(rdf[2]), float(-rdf[0]), float(-rdf[1]))
  np.testing.assert_allclose(mrc.rdf_to_world_flu(rdf), inline)


# --------------------------------------------------------------------------- #
# Nearest voxel within the probe radius
# --------------------------------------------------------------------------- #

def _rdf_grid(mrc, flu_points):
  return np.stack([mrc.world_flu_to_rdf(p) for p in flu_points])


def test_nearest_picks_the_closest_row(mrc):
  vox = _rdf_grid(mrc, [[0, 0, 0], [10, 0, 0], [-12.5, -33.0, 1.0]])
  k, d = mrc.nearest_index(vox, mrc.world_flu_to_rdf([-12.87, -33.52, 1.0]),
                           max_dist=3.0)
  assert k == 2
  assert d == pytest.approx(np.hypot(0.37, 0.52), abs=1e-9)


def test_nearest_rejects_beyond_the_radius_but_still_reports_the_distance(mrc):
  vox = _rdf_grid(mrc, [[0, 0, 0], [10, 10, 0]])
  k, d = mrc.nearest_index(vox, mrc.world_flu_to_rdf([0, 5, 0]), max_dist=3.0)
  assert k is None
  assert d == pytest.approx(5.0)


def test_nearest_on_an_empty_map(mrc):
  assert mrc.nearest_index(np.zeros((0, 3)), [0., 0., 0.], 3.0) == (
    None, float("inf"))
  assert mrc.nearest_index([], [0., 0., 0.], 3.0) == (None, float("inf"))


def test_a_2d_probe_point_ignores_height(mrc):
  """[x, y] constrains no altitude: a voxel 40 m up is still 'at' the point."""
  vox = _rdf_grid(mrc, [[0.0, 0.0, 40.0], [2.9, 0.0, 0.0]])
  p = (0.0, 0.0)
  k, d = mrc.nearest_index(vox, mrc.world_flu_to_rdf(p), max_dist=3.0,
                           axes=mrc.probe_axes(p))
  assert k == 0 and d == pytest.approx(0.0)


def test_a_3d_probe_point_does_not_ignore_height(mrc):
  vox = _rdf_grid(mrc, [[0.0, 0.0, 40.0], [2.9, 0.0, 0.0]])
  p = (0.0, 0.0, 0.0)
  k, d = mrc.nearest_index(vox, mrc.world_flu_to_rdf(p), max_dist=3.0,
                           axes=mrc.probe_axes(p))
  assert k == 1 and d == pytest.approx(2.9)


def test_rdf_horizontal_axes_are_the_non_vertical_ones(mrc):
  """RDF is (right, DOWN, forward), so the vertical column is 1."""
  assert mrc.RDF_HORIZONTAL_AXES == (0, 2)
  up_in_rdf = mrc.world_flu_to_rdf([0.0, 0.0, 1.0])
  assert abs(up_in_rdf[1]) == 1.0
  assert up_in_rdf[0] == 0.0 and up_in_rdf[2] == 0.0


def test_the_three_mission_points_resolve_to_distinct_voxels(mrc):
  """End to end on the exact values the mission yaml ships."""
  pts = mrc.parse_probe_points(MISSION_POINTS)
  # A 0.5 m voxel grid over the casualty patch, at an arbitrary height.
  gx, gy = np.meshgrid(np.arange(-20.0, 0.0, 0.5),
                       np.arange(-75.0, -25.0, 0.5))
  flu = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 1.25)], axis=1)
  vox = np.stack([mrc.world_flu_to_rdf(p) for p in flu])

  hits = []
  for p in pts:
    k, d = mrc.nearest_index(vox, mrc.world_flu_to_rdf(p), max_dist=3.0,
                             axes=mrc.probe_axes(p))
    assert k is not None, p
    assert d <= 0.5           # inside one voxel of the GT position
    hits.append(k)
    back = mrc.rdf_to_world_flu(vox[k])
    assert abs(back[0] - p[0]) <= 0.5 and abs(back[1] - p[1]) <= 0.5
  assert len(set(hits)) == 3
