"""Community AME-1 base course with native MuJoCo collision.

This is not a claim of Isaac/MuJoCo collision equivalence: the rough tile keeps
source height samples but uses MuJoCo's regular heightfield without the source
steep-edge XY correction. Its measured surface discrepancy is recorded beside
the experiment. Sparse tiles use a single convex decomposition of the corrected source surface
for sensing and contact. Exact cliff boundaries need one-sided validation.
"""
from dataclasses import dataclass
from functools import partial
import uuid

import mujoco
import numpy as np

from mjlab.terrains import BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput, TerrainGeneratorCfg

from .collision_pieces import add_collision_surface, collision_pieces
from .continuous import grid_boxes, pyramid_slope, rough
from .heightfields import SparseHeightfield, bordered_heightfield, concentric_gaps, stepping_stones
from .surface import convert_height_field_to_mesh


def add_heightfields(spec, body, heights):
    """Keep height samples in narrow strips, bounding G1's per-pair contacts."""
    physical = heights.T.astype(float)*.005
    minimum = float(physical.min())
    geometries = []
    # With the actual G1 collision radius <=.15m and .05m cells, two cell
    # rows keep candidate triangles below MuJoCo Warp's 50-contact pair cap.
    for start in range(0, physical.shape[0]-1, 2):
        stop = min(start+2, physical.shape[0]-1)
        patch = physical[start:stop+1]
        low = float(patch.min())
        span = max(float(np.ptp(patch)), .001)
        field = spec.add_hfield(name=f'community_hfield_{uuid.uuid4().hex}',
                               # A 10-micrometre overlap closes numerical ray
                               # holes at otherwise coincident strip boundaries.
                               size=[4, (stop-start)*.05/2+1e-5, span, .1+low-minimum],
                               nrow=patch.shape[0], ncol=patch.shape[1],
                               userdata=((patch-low)/span).ravel().tolist())
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                             pos=[4, (start+stop)*.05/2, low])
        geometries.append(TerrainGeometry(geom=geom, hfield=field))
    return geometries


def add_sparse_surface(spec, body, heights):
    vertices, triangles = convert_height_field_to_mesh(heights, .05, .005, .75)
    boxes, caps = collision_pieces(vertices, triangles, heights.shape)
    collision = add_collision_surface(spec, body, boxes, caps)
    return [TerrainGeometry(geom=geom) for geom in collision]


def build_tile(family, difficulty, spec, rng):
    body = spec.body('terrain')
    if family in ('stairs', 'inverse_stairs'):
        terrain_type = BoxPyramidStairsTerrainCfg if family == 'stairs' else BoxInvertedPyramidStairsTerrainCfg
        # Source platform_width=3 produces a physical 2.4 m platform here.
        return terrain_type(size=(8., 8.), border_width=1., platform_width=2.4,
                            step_width=.3, step_height_range=(.05, .2), holes=False).function(difficulty, spec, rng)
    if family == 'grid':
        boxes, origin = grid_boxes(difficulty, rng)
        geoms = [body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(low+high)/2,
                               size=(high-low)/2) for low, high in boxes]
        return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=g) for g in geoms])
    generators = dict(rough=rough, slope=pyramid_slope,
                      inverse_slope=partial(pyramid_slope, inverted=True),
                      stones=stepping_stones, gaps=concentric_gaps)
    heights, origin = bordered_heightfield(generators[family], difficulty, SparseHeightfield(), rng)
    if family in ('stones', 'gaps'):
        geometries = add_sparse_surface(spec, body, heights)
    else:
        geometries = add_heightfields(spec, body, heights)
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class CommunityColumn(SubTerrainCfg):
    family: str

    def function(self, difficulty, spec, rng):
        # The generator supplies row/9; source samples (row+U[0,1))/10.
        sampled_difficulty = (9*difficulty+rng.uniform())/10
        return build_tile(self.family, sampled_difficulty, spec, rng)


def make_community_terrain_generator():
    columns = {}
    for family, count in (('stairs', 2), ('inverse_stairs', 2), ('grid', 2),
                          ('rough', 2), ('slope', 2), ('inverse_slope', 2),
                          ('stones', 4), ('gaps', 4)):
        for index in range(count):
            columns[f'{family}_{index}'] = CommunityColumn(family=family)
    return TerrainGeneratorCfg(seed=42, curriculum=True, size=(8., 8.), num_rows=10,
                               num_cols=20, border_width=50., border_height=1.,
                               sub_terrains=columns, color_scheme='none')
