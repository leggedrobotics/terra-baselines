#!/usr/bin/env python3
"""Scripted admissible oracle for the fresh-trench dig-alignment gate.

The pilot's static preflight (``tools/audit_trench_alignment_feasibility.py``)
is a *static* witness: it shows a monotone all-or-nothing fresh-dig cover
exists over the 12x12 pose grid.  It does not show that a controller can
execute one from the actual episode spawn, inside the 450-step horizon, with
the real dig/dump/pile dynamics.

This script converts it into a *dynamic* witness.  It plays real
``TerraEnvBatch`` episodes on the frozen panel slots the pilot endpoint uses,
with ``enforce_trench_dig_alignment=True``, driven by a scripted (no-learning)
controller that reads privileged state:

  1. pick a gate-admissible dig station (base pose + cabin heading) that
     removes fresh trench soil and from which a legal *accepted* dump is
     available after the dig;
  2. navigate to it over a replica of Terra's pose graph;
  3. ``DO`` (dig), rotate the cabin to a legal accepted dump heading, ``DO``
     (dump);
  4. repeat until the episode terminates.

Two Terra runtime facts the offline preflight does not model are replicated
here and asserted against Terra itself:

  * ``State._build_dig_dump_cone`` is only *approximately* translation
    invariant (float32 boundary rounding), so every dig decision is validated
    with an exact per-pose Terra cone rather than an offset table;
  * ``terra.utils.compute_polygon_mask`` rasterises in ``[x=row, y=col]``
    order, matching ``pos_base``, so the pose-graph replica evaluates the
    footprint occupancy directly in pose space; ``--verify-action-mask`` checks
    it against ``State._get_action_mask_tracked`` every step.

The fresh-trench alignment validity replica is compared against the value
Terra exports in the observation on every step of every slot.  Any
disagreement raises.  So the reported completion rate is a *verified* lower
bound on what the episode contract admits, and the step counts are directly
comparable with the 450-step horizon.

GATE SEMANTICS.  The run measures **v2** by default (yaw-parallel only; working
distance is the dig cone's job, tested radially machine -> cell) and
``--gate-v1`` measures the retired v1 semantics, which additionally required the
perpendicular base-centre-to-axis standoff to sit in a 3.5-7.0 m lateral band.
The flag is forced onto the resolved env config and asserted, never inherited
from the checkpoint: v1 and v2 admit different stations, so a run that does not
say which one it measured is not comparable with anything.  Under v2 the
controller prefers ON-AXIS stations (perpendicular offset inside the retired
3.5 m floor -- dig ahead along the trench line, then retreat), which is the
station class v1 refused outright.

TWO GUARANTEES the controller enforces rather than hopes for:

  * it navigates over PERSISTENT poses only -- poses that stay legal with the
    WHOLE trench dug (``fp_conservative``) -- with a first-leg exception out of
    the spawn pocket, so it can never wall itself off behind its own
    excavation.  Both the pose-graph BFS and the station set use that
    restriction (``pose_bfs(persist=...)``).
  * every station is an airtight dig+DUMP PAIR: the post-dig state is predicted
    exactly (Terra's own dig selection, lowered by ``dig_depth``, written into
    ``last_dig_mask``) and a cabin with a legal ACCEPTED dump must exist in it,
    or the station is refused.  A dump is additionally re-validated against
    Terra's live cone in the step that executes it and is never pressed without
    a legal accepted destination -- Terra's fallback branch would drop the load
    outside ``target > 0``, which is unrecoverable spoil.  An excavator cannot
    move while loaded, so a dig without a dump is a deadlock, not a setback.

No policy is used; the checkpoint is opened read-only only to inherit the
treatment arm's exact ``train_config`` (hence its env config and gate flag).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_fixed_bank import (  # noqa: E402
    configure_for_bank,
    exact_reset_keys,
    load_manifest,
    manifest_environment_keys,
    prepare_manifest_episode_reset,
)
from train import TrainConfig  # noqa: E402
from train_mixed import MixedAgentTrainConfig, make_mixed_agent_states  # noqa: E402
from utils.accepted_bank import load_accepted_bank, V8_RELEASE_ID  # noqa: E402
from utils.helpers import load_pkl_object  # noqa: E402
from utils.utils_ppo import wrap_action  # noqa: E402

sys.modules["__main__"].TrainConfig = TrainConfig
sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

FORWARD, BACKWARD, CLOCK, ANTICLOCK, CABIN_CLOCK, CABIN_ANTICLOCK, DO, DO_NOTHING = range(8)
ACTION_NAMES = ["FORWARD", "BACKWARD", "CLOCK", "ANTICLOCK", "CABIN_CLOCK",
                "CABIN_ANTICLOCK", "DO", "DO_NOTHING"]
NH = 12
SHAPE = (64, 64)
INF = np.int32(1 << 20)
SCHEMA = "terra_trench_align_scripted_oracle_v1"


# ---------------------------------------------------------------------------
# Terra geometry, taken from Terra itself
# ---------------------------------------------------------------------------


class Geometry:
    """Cone / footprint / move tables plus exact per-pose Terra cone access."""

    def __init__(self, env_cfg_single):
        from terra.state import State
        from terra.utils import compute_polygon_mask

        self.cfg = env_cfg_single
        state = State.new(
            jax.random.PRNGKey(0),
            env_cfg_single,
            np.zeros(SHAPE, dtype=np.int8),
            np.zeros(SHAPE, dtype=np.int8),
            -97.0 * np.ones((4, 8), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            np.ones(SHAPE, dtype=np.bool_),
            np.zeros(SHAPE, dtype=np.int8),
            distance_map_override=np.ones(SHAPE, dtype=np.float32),
        )
        self._state = state

        def _pose(row, col, bh, cb):
            cur = state._get_current_agent_state()._replace(
                pos_base=jnp.stack([row, col]).astype(jnp.int16),
                angle_base=jnp.reshape(bh, (1,)).astype(jnp.int8),
                angle_cabin=jnp.reshape(cb, (1,)).astype(jnp.int8),
                loaded=jnp.zeros((1,), dtype=jnp.int8),
            )
            return state._set_current_agent_state(cur)

        self._cone_one = jax.jit(
            lambda r, c, b, k: _pose(r, c, b, k)._build_dig_dump_cone().reshape(SHAPE)
        )
        self._cone_vec = jax.jit(
            jax.vmap(
                lambda r, c, b, k: _pose(r, c, b, k)._build_dig_dump_cone().reshape(SHAPE)
            )
        )

        center = np.array([32, 32])
        self.cones = []
        self.fwd = []
        self.bwd = []
        for bh in range(NH):
            row = []
            for cb in range(NH):
                mask = np.asarray(
                    self._cone_one(jnp.int32(32), jnp.int32(32), jnp.int32(bh), jnp.int32(cb))
                )
                row.append((np.argwhere(mask) - center).astype(np.int32))
            self.cones.append(row)
            p = _pose(jnp.int32(32), jnp.int32(32), jnp.int32(bh), jnp.int32(0))
            f = np.asarray(p._handle_move_forward()._get_current_agent_state().pos_base)
            b = np.asarray(p._handle_move_backward()._get_current_agent_state().pos_base)
            self.fwd.append(tuple(int(v) for v in (f.reshape(-1) - center)))
            self.bwd.append(tuple(int(v) for v in (b.reshape(-1) - center)))

        self._build_footprint_tables(env_cfg_single)
        self._build_move_tables(env_cfg_single)

    def _build_move_tables(self, cfg):
        """Exact per-pose move destinations.

        ``State._move_on_orientation`` computes ``round(pos_base + delta_xy)``
        in float32 with round-half-to-even, and half of the 12 headings have a
        delta component of exactly +/-2.5 tiles, so the integer step alternates
        between 2 and 3 with the parity of the coordinate.  A single per-heading
        delta is therefore wrong for half the poses.
        """
        # xy_delta is taken from Terra's own expression, in JAX float32, so the
        # round-half-to-even step lands on the same integer.
        angles = jnp.linspace(0, 2 * jnp.pi, NH, endpoint=False)
        angles = (angles + (jnp.pi / 2)) % (2 * jnp.pi)
        xy_delta = int(cfg.agent.move_tiles) * jnp.stack(
            [jnp.cos(angles), jnp.sin(angles)], axis=-1
        )
        grid = jnp.stack(jnp.meshgrid(
            jnp.arange(SHAPE[0], dtype=jnp.int16),
            jnp.arange(SHAPE[1], dtype=jnp.int16), indexing="ij"), axis=-1)
        self.succ_flat = np.full((NH, 2, SHAPE[0] * SHAPE[1]), -1, dtype=np.int32)
        self.pred_lists = [[[] for _ in range(SHAPE[0] * SHAPE[1])] for _ in range(NH)]
        for bh in range(NH):
            for a, src_bh in enumerate((bh, (bh + NH // 2) % NH)):
                cand = np.asarray(
                    jnp.round(grid + xy_delta[src_bh]).astype(jnp.int32)
                )
                RR, CC = cand[:, :, 0], cand[:, :, 1]
                ok = (RR >= 0) & (RR < SHAPE[0]) & (CC >= 0) & (CC < SHAPE[1])
                flat = np.where(ok, RR * SHAPE[1] + CC, -1).reshape(-1)
                self.succ_flat[bh, a] = flat
                act = FORWARD if a == 0 else BACKWARD
                src = np.flatnonzero(flat >= 0)
                for i in src:
                    self.pred_lists[bh][int(flat[i])].append((int(i), act))

    # -- exact per-pose footprint -----------------------------------------
    def _build_footprint_tables(self, cfg):
        """Exact footprint occupancy per pose.

        ``State._get_agent_corners`` rounds ``R @ local + pos`` with a
        centre-biased floor/ceil in float32, so for heading 6 (sin == -1.2e-16)
        the integer corner offsets depend on the pose.  Corner offsets are
        therefore computed exactly per pose and grouped: rows only affect the
        x-coordinates and columns only the y-coordinates, so the number of
        distinct patterns is tiny.
        """
        f32 = np.float32
        aw = f32(int(cfg.agent.width))
        ah = f32(int(cfg.agent.height))
        local = np.array([
            [-np.floor(aw / 2.0), -np.floor(ah / 2.0)],
            [np.ceil(aw / 2.0), -np.floor(ah / 2.0)],
            [np.ceil(aw / 2.0), np.ceil(ah / 2.0)],
            [-np.floor(aw / 2.0), np.ceil(ah / 2.0)],
        ], dtype=np.float32)
        idx = np.arange(SHAPE[0], dtype=np.float32)

        self.fp_pattern_id = np.zeros((NH,) + SHAPE, dtype=np.int32)
        self.fp_pattern_offsets = []   # [bh][pattern] -> (K,2) mask-space offsets
        self.bounds_ok = np.zeros((NH,) + SHAPE, dtype=bool)
        for bh in range(NH):
            ang = f32(f32(bh) / f32(NH) * f32(2.0 * np.pi))
            cos_a, sin_a = np.cos(ang, dtype=np.float32), np.sin(ang, dtype=np.float32)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            rot = (R @ local.T).T.astype(np.float32)          # (4,2)
            gx = (rot[None, :, 0] + idx[:, None]).astype(np.float32)   # (64,4) per row
            gy = (rot[None, :, 1] + idx[:, None]).astype(np.float32)   # (64,4) per col
            bx = np.where(gx < idx[:, None], np.floor(gx), np.ceil(gx)).astype(np.int32)
            by = np.where(gy < idx[:, None], np.floor(gy), np.ceil(gy)).astype(np.int32)
            dx = bx - np.arange(SHAPE[0])[:, None]            # (64,4) row offsets
            dy = by - np.arange(SHAPE[1])[:, None]            # (64,4) col offsets
            row_key = {}
            col_key = {}
            row_id = np.zeros(SHAPE[0], dtype=np.int32)
            col_id = np.zeros(SHAPE[1], dtype=np.int32)
            for i in range(SHAPE[0]):
                row_id[i] = row_key.setdefault(tuple(dx[i].tolist()), len(row_key))
            for j in range(SHAPE[1]):
                col_id[j] = col_key.setdefault(tuple(dy[j].tolist()), len(col_key))
            row_patterns = [np.array(k, dtype=np.int32) for k, _ in
                            sorted(row_key.items(), key=lambda kv: kv[1])]
            col_patterns = [np.array(k, dtype=np.int32) for k, _ in
                            sorted(col_key.items(), key=lambda kv: kv[1])]
            n_col = len(col_patterns)
            self.fp_pattern_id[bh] = row_id[:, None] * n_col + col_id[None, :]
            offsets = []
            for rp in row_patterns:
                for cp in col_patterns:
                    corners = np.stack([rp + 32, cp + 32], axis=1)
                    mask = polygon_mask_np(corners.astype(np.float32))
                    offsets.append((np.argwhere(mask) - np.array([32, 32])).astype(np.int32))
            self.fp_pattern_offsets.append(offsets)
            # bounds: true corners must lie in [0, 64)
            ok = np.ones(SHAPE, dtype=bool)
            for corner in range(4):
                r_ok = (bx[:, corner] >= 0) & (bx[:, corner] < SHAPE[0])
                c_ok = (by[:, corner] >= 0) & (by[:, corner] < SHAPE[1])
                ok &= r_ok[:, None] & c_ok[None, :]
            self.bounds_ok[bh] = ok

    def cone_at(self, row, col, bh, cb):
        return np.asarray(
            self._cone_one(jnp.int32(row), jnp.int32(col), jnp.int32(bh), jnp.int32(cb))
        )

    def cones_all_cabins(self, row, col, bh):
        r = jnp.full((NH,), row, dtype=jnp.int32)
        c = jnp.full((NH,), col, dtype=jnp.int32)
        b = jnp.full((NH,), bh, dtype=jnp.int32)
        k = jnp.arange(NH, dtype=jnp.int32)
        return np.asarray(self._cone_vec(r, c, b, k))

    def cones_batch(self, rows, cols, bhs, cbs):
        return np.asarray(
            self._cone_vec(
                jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(cols, dtype=jnp.int32),
                jnp.asarray(bhs, dtype=jnp.int32), jnp.asarray(cbs, dtype=jnp.int32),
            )
        )

    def verify_translation(self):
        """Measure how exactly the cone offset table transfers, and verify the
        per-pose footprint occupancy against Terra."""
        from terra.utils import compute_polygon_mask

        diffs = []
        fp_bad = 0
        fp_checks = 0
        for row, col in ((18, 41), (28, 21), (44, 33), (12, 12), (8, 55), (55, 8)):
            for bh in range(NH):
                for cb in (0, 3, 6, 9):
                    got = self.cone_at(row, col, bh, cb).astype(bool)
                    want = np.zeros(SHAPE, dtype=bool)
                    cells = self.cones[bh][cb] + np.array([row, col])
                    ins = (
                        (cells[:, 0] >= 0) & (cells[:, 0] < SHAPE[0])
                        & (cells[:, 1] >= 0) & (cells[:, 1] < SHAPE[1])
                    )
                    cells = cells[ins]
                    want[cells[:, 0], cells[:, 1]] = True
                    diffs.append(int(np.sum(got ^ want)))
                cur = self._state._get_current_agent_state()._replace(
                    pos_base=jnp.array([row, col], dtype=jnp.int16),
                    angle_base=jnp.array([bh], dtype=jnp.int8),
                    angle_cabin=jnp.zeros((1,), dtype=jnp.int8),
                    loaded=jnp.zeros((1,), dtype=jnp.int8),
                )
                s2 = self._state._set_current_agent_state(cur)
                corners = s2._get_agent_corners(
                    cur.pos_base, base_orientation=cur.angle_base,
                    agent_width=self.cfg.agent.width, agent_height=self.cfg.agent.height,
                )
                got_fp = np.asarray(compute_polygon_mask(corners, 64, 64)).astype(bool)
                want_fp = np.zeros(SHAPE, dtype=bool)
                cells = self.footprint_offsets(bh, row, col) + np.array([row, col])
                ins = (
                    (cells[:, 0] >= 0) & (cells[:, 0] < SHAPE[0])
                    & (cells[:, 1] >= 0) & (cells[:, 1] < SHAPE[1])
                )
                cells = cells[ins]
                want_fp[cells[:, 0], cells[:, 1]] = True
                fp_checks += 1
                if not np.array_equal(got_fp, want_fp):
                    fp_bad += 1
        if fp_bad:
            raise RuntimeError(f"footprint replica wrong at {fp_bad}/{fp_checks} poses")

        # the vmapped cone accessors must agree with the single-pose one
        vec_checks = 0
        probe_poses = [(20, 20, 3), (32, 32, 0), (41, 18, 5), (15, 45, 9), (12, 51, 6)]
        for row, col, bh in probe_poses:
            allc = self.cones_all_cabins(row, col, bh)
            for cb in range(NH):
                one = self.cone_at(row, col, bh, cb).astype(bool)
                vec_checks += 1
                if not np.array_equal(allc[cb].astype(bool), one):
                    raise RuntimeError(
                        f"cones_all_cabins disagrees with cone_at at "
                        f"({row},{col}) bh={bh} cb={cb}: "
                        f"{int(allc[cb].sum())} vs {int(one.sum())}")
        rows_b = np.array([p[0] for p in probe_poses])
        cols_b = np.array([p[1] for p in probe_poses])
        bhs_b = np.array([p[2] for p in probe_poses])
        for cb in range(NH):
            batch = self.cones_batch(rows_b, cols_b, bhs_b,
                                     np.full(len(probe_poses), cb))
            for i, (row, col, bh) in enumerate(probe_poses):
                one = self.cone_at(row, col, bh, cb).astype(bool)
                vec_checks += 1
                if not np.array_equal(batch[i].astype(bool), one):
                    raise RuntimeError(
                        f"cones_batch disagrees with cone_at at ({row},{col}) "
                        f"bh={bh} cb={cb}: {int(batch[i].sum())} vs {int(one.sum())}")

        # move destinations, against Terra on an obstacle-free map
        move_checks = 0
        base = self._state

        def _moved(row, col, bh, a):
            cur = base._get_current_agent_state()._replace(
                pos_base=jnp.stack([row, col]).astype(jnp.int16),
                angle_base=jnp.reshape(bh, (1,)).astype(jnp.int8),
                angle_cabin=jnp.zeros((1,), dtype=jnp.int8),
                loaded=jnp.zeros((1,), dtype=jnp.int8),
            )
            st2 = base._set_current_agent_state(cur)
            return jax.lax.cond(
                a == 0,
                lambda: st2._handle_move_forward()._get_current_agent_state().pos_base,
                lambda: st2._handle_move_backward()._get_current_agent_state().pos_base,
            )

        moved = jax.jit(jax.vmap(_moved))
        rows_v, cols_v, bh_v, a_v = [], [], [], []
        for row in range(6, 58, 5):
            for col in range(6, 58, 7):
                for bh in range(NH):
                    for a in range(2):
                        rows_v.append(row); cols_v.append(col)
                        bh_v.append(bh); a_v.append(a)
        got_all = np.asarray(moved(
            jnp.asarray(rows_v, dtype=jnp.int32), jnp.asarray(cols_v, dtype=jnp.int32),
            jnp.asarray(bh_v, dtype=jnp.int32), jnp.asarray(a_v, dtype=jnp.int32),
        )).reshape(len(rows_v), 2)
        for t in range(len(rows_v)):
            row, col, bh, a = rows_v[t], cols_v[t], bh_v[t], a_v[t]
            if True:
                if True:
                    for _ in (0,):
                        got = got_all[t]
                        want = int(self.succ_flat[bh, a][row * SHAPE[1] + col])
                        if want < 0:
                            continue  # out of bounds: Terra refuses, position unchanged
                        wr, wc = want // SHAPE[1], want % SHAPE[1]
                        if int(got[0]) == row and int(got[1]) == col:
                            continue  # Terra refused the move (bounds/occupancy)
                        move_checks += 1
                        if int(got[0]) != wr or int(got[1]) != wc:
                            raise RuntimeError(
                                f"move table wrong at ({row},{col}) bh={bh} a={a}: "
                                f"Terra {tuple(int(v) for v in got)} replica {(wr, wc)}"
                            )
        diffs = np.asarray(diffs)
        return {
            "cone_offset_table_checks": int(diffs.size),
            "cone_offset_table_exact_fraction": float((diffs == 0).mean()),
            "cone_offset_table_mean_cell_diff": float(diffs.mean()),
            "cone_offset_table_max_cell_diff": int(diffs.max()),
            "footprint_replica_checks": fp_checks,
            "footprint_patterns_per_heading": [len(x) for x in self.fp_pattern_offsets],
            "move_table_checks": move_checks,
            "cone_accessor_checks": vec_checks,
        }

    def footprint_offsets(self, bh, row, col):
        return self.fp_pattern_offsets[bh][int(self.fp_pattern_id[bh, row, col])]


# ---------------------------------------------------------------------------
# numpy replicas
# ---------------------------------------------------------------------------


def blocked_mask(action_map: np.ndarray, static_base: np.ndarray) -> np.ndarray:
    """Replica of ``State._build_traversability_mask`` (True == blocked)."""
    static = static_base.astype(bool)
    if not np.any(action_map != 0):
        return static
    dirt = (action_map != 0).astype(np.int32)
    padded = np.pad(dirt, 1)
    cnt = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + dirt + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    return static | ((action_map != 0) & (cnt >= 6)) | (action_map > 1) | (action_map < 0)


def polygon_mask_np(corners: np.ndarray) -> np.ndarray:
    """float32 replica of ``terra.utils.compute_polygon_mask`` (64x64)."""
    xs = np.arange(SHAPE[0], dtype=np.float32) + np.float32(0.5)
    ys = np.arange(SHAPE[1], dtype=np.float32) + np.float32(0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X, Y], axis=-1).reshape((-1, 2)).astype(np.float32)
    edges = (np.roll(corners, -1, axis=0) - corners).astype(np.float32)
    diff = (pts[None, :, :] - corners[:, None, :]).astype(np.float32)
    cross = edges[:, None, 0] * diff[..., 1] - edges[:, None, 1] * diff[..., 0]
    inside = np.logical_or(np.all(cross > 0, axis=0), np.all(cross < 0, axis=0))
    return inside.reshape(SHAPE)


def footprint_free(blocked: np.ndarray, geom: Geometry) -> np.ndarray:
    """(12, 64, 64) bool: is pose (bh, row, col) a legal Terra pose?

    ``terra.utils.compute_polygon_mask`` rasterises in ``[x=row, y=col]`` order,
    matching ``pos_base``, so the occupancy correlation is evaluated directly in
    pose space.  Bounds use the exact per-pose corner extents.
    """
    out = np.zeros((NH,) + SHAPE, dtype=bool)
    pad = 40
    big = np.ones((SHAPE[0] + 2 * pad, SHAPE[1] + 2 * pad), dtype=bool)
    big[pad:pad + SHAPE[0], pad:pad + SHAPE[1]] = blocked
    for bh in range(NH):
        pids = geom.fp_pattern_id[bh]
        free_pose = np.zeros(SHAPE, dtype=bool)
        for pid, offs in enumerate(geom.fp_pattern_offsets[bh]):
            sel = pids == pid
            if not sel.any():
                continue
            acc = np.zeros(SHAPE, dtype=bool)
            for da, db in offs:
                acc |= big[pad + da:pad + da + SHAPE[0], pad + db:pad + db + SHAPE[1]]
            free_pose |= (~acc) & sel
        out[bh] = free_pose & geom.bounds_ok[bh]
    return out


def _bfs_level(dist, allowed, frontier, step, geom):
    """One synchronous BFS level over the pose graph. ``frontier`` maps h -> idx."""
    buckets = {}

    def push(h, cand):
        if cand.size == 0:
            return
        cand = cand[allowed[h, cand] & (dist[h, cand] == INF)]
        if cand.size == 0:
            return
        cand = np.unique(cand)
        dist[h, cand] = step
        buckets.setdefault(h, []).append(cand)

    for bh, idxs in frontier.items():
        push((bh - 1) % NH, idxs)
        push((bh + 1) % NH, idxs)
        for a in range(2):
            d = geom.succ_flat[bh, a][idxs]
            push(bh, d[d >= 0])
    return {h: np.concatenate(v) for h, v in buckets.items()}


def pose_bfs(start, fp_free, geom, persist=None):
    """Undirected BFS over Terra's exact pose graph. Returns (12, 64, 64) int32.

    With ``persist`` given the walk is restricted to PERSISTENT poses -- poses
    that stay legal even when the whole trench is dug -- so the controller can
    never wall itself off with its own excavation.  The spawn is exempt: a
    machine that starts inside the trench footprint has to be allowed to drive
    out of it.  So the search first escapes the non-persistent pocket it starts
    in (phase 1, over ``fp_free``, absorbing at the first persistent pose on
    each branch) and is persistent-only from there on (phase 2).  The result is
    the shortest distance under "non-persistent prefix, then persistent
    suffix", and every dig station is taken from the persistent set.
    """
    n = SHAPE[0] * SHAPE[1]
    dist = np.full((NH, n), INF, dtype=np.int32)
    free = fp_free.reshape(NH, n)
    keep = free if persist is None else (fp_free & persist).reshape(NH, n)
    r0, c0, h0 = start
    idx0 = r0 * SHAPE[1] + c0
    dist[h0, idx0] = 0

    seeds: dict[int, dict[int, list]] = {}
    if keep[h0, idx0]:
        frontier = {h0: np.array([idx0], dtype=np.int32)}
    else:
        frontier = {h0: np.array([idx0], dtype=np.int32)}
        step = 0
        while frontier:
            step += 1
            produced = _bfs_level(dist, free, frontier, step, geom)
            nxt = {}
            for h, idxs in produced.items():
                pers = keep[h, idxs]
                if pers.any():
                    seeds.setdefault(step, {}).setdefault(h, []).append(idxs[pers])
                rest = idxs[~pers]
                if rest.size:
                    nxt[h] = rest
            frontier = nxt
        frontier = {}

    level = 0
    last_seed = max(seeds) if seeds else 0
    while frontier or level < last_seed:
        if level in seeds:
            for h, parts in seeds[level].items():
                arr = np.concatenate(parts)
                frontier[h] = (
                    np.concatenate([frontier[h], arr]) if h in frontier else arr
                )
        if frontier:
            frontier = _bfs_level(dist, keep, frontier, level + 1, geom)
        level += 1
    return dist.reshape((NH,) + SHAPE)


def reconstruct_actions(dist, goal, geom, persist=None):
    """Descend ``dist`` to the start, preferring predecessors of the same class.

    With ``persist`` given, a persistent node keeps a persistent predecessor
    whenever one exists; the first time it cannot, the walk has crossed into the
    spawn's non-persistent pocket and stays there.  That keeps the executed path
    "pocket prefix, then persistent suffix" rather than re-entering a pose the
    machine's own digging could delete.
    """
    r, c, h = goal
    if dist[h, r, c] >= INF:
        return None
    actions = []
    guard = 0
    idx = r * SHAPE[1] + c
    flat = dist.reshape(NH, -1)
    pers = None if persist is None else persist.reshape(NH, -1)
    want_persistent = pers is not None and bool(pers[h, idx])
    while flat[h, idx] > 0:
        guard += 1
        if guard > 4096:
            return None
        d = flat[h, idx]
        # candidate predecessors: (heading, index, action)
        options = []
        for dh, act in ((-1, ANTICLOCK), (1, CLOCK)):
            ph = (h + dh) % NH
            if flat[ph, idx] == d - 1:
                options.append((ph, idx, act))
        for pidx, act in geom.pred_lists[h][idx]:
            if flat[h, pidx] == d - 1:
                options.append((h, pidx, act))
        if not options:
            return None
        choice = None
        if pers is not None and want_persistent:
            for ph, pidx, act in options:
                if pers[ph, pidx]:
                    choice = (ph, pidx, act)
                    break
            if choice is None:
                want_persistent = False
        if choice is None:
            choice = options[0]
        h, idx, act = choice
        actions.append(act)
    actions.reverse()
    return actions


def cabin_actions(cur_cb, want_cb):
    delta = (want_cb - cur_cb) % NH
    if delta <= NH - delta:
        return [CABIN_ANTICLOCK] * delta
    return [CABIN_CLOCK] * (NH - delta)


def gather_counts(poses, offs, maps):
    rr = poses[:, 0:1] + offs[None, :, 0]
    cc = poses[:, 1:2] + offs[None, :, 1]
    ok = (rr >= 0) & (rr < SHAPE[0]) & (cc >= 0) & (cc < SHAPE[1])
    rrc = np.clip(rr, 0, SHAPE[0] - 1)
    ccc = np.clip(cc, 0, SHAPE[1] - 1)
    out = np.empty((len(maps), poses.shape[0]), dtype=np.int32)
    for i, m in enumerate(maps):
        out[i] = np.sum(m[rrc, ccc] & ok, axis=1)
    return out


def dilate_linf(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def dilate_square5(mask: np.ndarray) -> np.ndarray:
    """5x5 square dilation, matching ``compute_dynamic_dumpability``."""
    pad = np.zeros((SHAPE[0] + 4, SHAPE[1] + 4), dtype=bool)
    pad[2:-2, 2:-2] = mask
    out = np.zeros(SHAPE, dtype=bool)
    for dr in range(5):
        for dc in range(5):
            out |= pad[dr:dr + SHAPE[0], dc:dc + SHAPE[1]]
    return out


class SlotOracle:
    IDLE, NAV, DUMP = 0, 1, 2

    def __init__(self, index, cell, target, padding, dumpability_init, records,
                 naxes, membership, geom, tile_size, yaw_tol, so_min, so_max,
                 standoff_enforced, dig_depth):
        self.index = index
        self.cell = cell
        self.target = target.astype(np.int32)
        self.padding = padding.astype(bool)
        self.dumpability_init = dumpability_init.astype(bool)
        self.records = records
        self.naxes = int(naxes)
        self.membership = membership
        self.geom = geom
        self.accepted = (self.target > 0) & (~self.padding)
        # Accepted cells whose 3x3 neighbourhood is also accepted.  Terra's dump
        # commits only when soil relaxation stays inside the containment mask
        # (the accepted zone), so dumping into the interior is what keeps a
        # dump from silently no-opping.
        self.accepted_interior = self.accepted & ~dilate_linf(~self.accepted, 1)
        self.dig_targets = self.target < 0

        # float32 throughout, matching State._get_fresh_trench_dig_alignment_details
        axes = records[: self.naxes, :3].astype(np.float32)
        denom = np.maximum(np.linalg.norm(axes[:, :2], axis=1), np.float32(1e-6))
        rows, cols = np.meshgrid(
            np.arange(SHAPE[0], dtype=np.float32),
            np.arange(SHAPE[1], dtype=np.float32),
            indexing="ij",
        )
        self.standoff = np.empty((self.naxes,) + SHAPE, dtype=np.float32)
        for a in range(self.naxes):
            self.standoff[a] = (
                np.abs(axes[a, 0] * cols + axes[a, 1] * rows + axes[a, 2]) / denom[a]
            ).astype(np.float32) * np.float32(tile_size)
        self.band = (self.standoff >= np.float32(so_min)) & (self.standoff <= np.float32(so_max))

        tangents = np.stack([-axes[:, 0], axes[:, 1]], axis=1).astype(np.float32)
        tnorm = np.maximum(np.linalg.norm(tangents, axis=1), np.float32(1e-6))
        self.yaw_ok = np.zeros((NH, self.naxes), dtype=bool)
        self.yaw_err = np.zeros((NH, self.naxes), dtype=np.float32)
        for bh in range(NH):
            theta = np.float32(2.0 * np.pi) * np.float32(bh) / np.float32(NH)
            forward = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
            cosine = np.clip((np.abs(tangents @ forward) / tnorm).astype(np.float32),
                             np.float32(0.0), np.float32(1.0))
            self.yaw_err[bh] = np.arccos(cosine)
            self.yaw_ok[bh] = self.yaw_err[bh] <= np.float32(yaw_tol)

        # Pose validity per (section, heading, cell), exactly as the gate tests
        # it.  v1 (standoff_enforced): yaw-parallel AND inside the lateral
        # standoff band.  v2 (the default): yaw-parallel, full stop -- working
        # distance is the dig cone's job and it is tested radially,
        # machine -> cell, so standing ON the trench line is legal.
        self.standoff_enforced = bool(standoff_enforced)
        self.dig_depth = int(dig_depth)
        self.pose_ok = np.zeros((max(self.naxes, 1), NH) + SHAPE, dtype=bool)
        for a in range(self.naxes):
            for bh in range(NH):
                if not self.yaw_ok[bh, a]:
                    continue
                self.pose_ok[a, bh] = self.band[a] if self.standoff_enforced else True
        # "on axis" = nearer the section line than the retired v1 floor: the
        # dig-ahead-retreat station v1 refused and v2 admits.
        self.on_axis = self.standoff < np.float32(so_min)

        # Order-independent worst case: a pose that stays legal with the whole
        # trench dug can never be walled off by the machine's own digging.
        self.fp_conservative = footprint_free(self.padding | self.dig_targets, geom)

        self.phase = self.IDLE
        self.plan = []
        self.goal = None
        self.dump_cabin = None
        self.stations = 0
        self.digs = 0
        self.dumps = 0
        self.failed_digs = 0
        self.failed_dumps = 0
        self.replans = 0
        self.no_candidate = 0
        self.stuck_loaded = 0
        self.move_refused = 0
        self.blocked_until_change = False
        self.stall_attribution = None
        self.stations_on_axis = 0
        self.dump_precheck_blocks = 0
        self.reasons = {}

    def note(self, reason):
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def pose_bits(self, r, c, bh):
        bits = 0
        for a in range(self.naxes):
            if self.pose_ok[a, bh, r, c]:
                bits |= 1 << a
        return bits

    def station_is_on_axis(self, r, c, bh):
        """Is this a dig-ahead-retreat station (inside the retired v1 floor)?"""
        return any(
            bool(self.pose_ok[a, bh, r, c]) and bool(self.on_axis[a, r, c])
            for a in range(self.naxes)
        )

    def dig_admissible(self, r, c, bh, cone, action_map, last_dig):
        """Exact replica of applicability + gate validity for one prospective DO.

        Returns (applicable, valid, selected_cells, fresh_trench_cells,
        selected_mask).  The mask is Terra's own dig selection, which is what
        ``_handle_dig`` removes and what it writes into ``last_dig_mask``, so
        the dump side can be predicted from it exactly rather than guessed.
        """
        selm = np.zeros(SHAPE, dtype=bool)
        rr, cc = np.nonzero(cone)
        if rr.size == 0:
            return False, True, 0, 0, selm
        am = action_map[rr, cc]
        tg = self.target[rr, cc]
        has_pile = bool(np.any(am > 0))
        amb = (am > 0) if has_pile else (am == 0)
        sel = ((tg < 0) | (am > 0)) & amb & (am > -1) & (~last_dig[rr, cc])
        selm[rr[sel], cc[sel]] = True
        fresh = sel & (tg < 0) & (am == 0)
        mem = self.membership[rr, cc]
        fresh_trench = fresh & (mem != 0)
        n_fresh_trench = int(fresh_trench.sum())
        if n_fresh_trench == 0:
            return False, True, int(sel.sum()), 0, selm
        bits = self.pose_bits(r, c, bh)
        valid = bool(np.all((mem[fresh_trench] & np.uint8(bits)) != 0))
        return True, valid, int(sel.sum()), n_fresh_trench, selm

    def dump_context(self, action_map, occupied):
        """Everything in the dump mask that does not depend on the cabin.

        ``dilate_square5`` over the whole hole map is the expensive part and it
        is identical for all twelve cabin headings, so it is computed once per
        (map, base pose) and reused -- the dump search calls the mask twelve
        times per candidate station.
        """
        holes = action_map < 0
        dumpable = self.dumpability_init & ~dilate_square5(holes)
        free = (action_map == 0) & (~occupied) & (~self.padding)
        return holes, ((~holes) & dumpable & free & self.accepted)

    def dump_legal_mask(self, cone, action_map, last_dig, occupied, ctx=None):
        """Replica of the excavator dump mask's legal-accepted branch.

        Terra dumps into ``accepted`` (target > 0) whenever ANY accepted cell
        survives the physical filters, and falls back to non-accepted cells only
        when none does -- the one path that creates illegal spoil.  So a
        non-empty mask here is exactly the condition "this dump lands inside the
        zone", and the controller never presses DO while loaded without it.
        """
        _holes, legal_anywhere = (
            self.dump_context(action_map, occupied) if ctx is None else ctx
        )
        rr, cc = np.nonzero(cone)
        if rr.size == 0:
            return np.zeros(SHAPE, dtype=bool)
        if np.any(last_dig[rr, cc] & (action_map[rr, cc] > 0)):
            return np.zeros(SHAPE, dtype=bool)
        return cone & legal_anywhere

    def dump_legal_count(self, cone, action_map, last_dig, occupied, ctx=None):
        return int(
            self.dump_legal_mask(cone, action_map, last_dig, occupied, ctx).sum()
        )

    def occupied_mask(self, r, c, bh):
        occ = np.zeros(SHAPE, dtype=bool)
        cells = self.geom.footprint_offsets(bh, r, c) + np.array([r, c])
        ins = (
            (cells[:, 0] >= 0) & (cells[:, 0] < SHAPE[0])
            & (cells[:, 1] >= 0) & (cells[:, 1] < SHAPE[1])
        )
        cells = cells[ins]
        occ[cells[:, 0], cells[:, 1]] = True
        return occ


def _dump_cabin_after_dig(o, r2, c2, bh2, cb2, cones12, action_map, last_dig,
                          sel_mask, keep_clear=None):
    """Cabin heading with a legal accepted dump once the dig at cb2 has happened.

    This is the DUMP half of the dig+dump pair, and it is what makes a station
    choice airtight: the machine cannot move while loaded, so a dig with no
    same-base legal dump is a hard deadlock.  The post-dig state is predicted
    EXACTLY -- ``sel_mask`` is Terra's own dig selection, ``_handle_dig`` lowers
    exactly those cells by ``dig_depth`` and writes exactly that mask into
    ``last_dig_mask`` -- so the dump legality tested here is the dump legality
    that will hold after the dig.  Returns None when no cabin has one, i.e.
    "do not take this station".

    Preferences among the cabins that do work, in order:
      * ``far``: no legal cell lies within ``keep_clear`` of still-undug trench.
        A pile next to fresh trench makes ``_mask_out_wrong_dig_tiles`` select
        the pile instead of the soil and blocks digging there.
      * ``interior``: every legal cell's 3x3 neighbourhood is inside the
        accepted zone.  Terra refuses the whole dump (``stayed_contained``) if
        soil relaxation would push a unit outside the containment mask, so an
        interior target is the one least likely to no-op.
      * otherwise the maximum-capacity cabin.
    """
    pred = action_map.copy()
    pred[sel_mask] = action_map[sel_mask] - o.dig_depth
    occ = o.occupied_mask(r2, c2, bh2)
    ctx = o.dump_context(pred, occ)
    ranked = []
    for cbd in range(NH):
        cone_d = cones12[cbd].astype(bool)
        legal = o.dump_legal_mask(cone_d, pred, sel_mask, occ, ctx)
        n = int(legal.sum())
        if n == 0:
            continue
        far = keep_clear is None or not bool(np.any(legal & keep_clear))
        interior = not bool(np.any(legal & ~o.accepted_interior))
        ranked.append((int(far) * 2 + int(interior), n, cbd))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (-t[0], -t[1]))
    return ranked[0][2]


ON_AXIS_BONUS = 6
POSE_CAP = 400
SWEEP_CAP = 150


def choose_station(o: SlotOracle, r, c, bh, cb, action_map, last_dig, banned,
                   fp_free, dist, max_probe=40, reach=12, persistent_only=True):
    """Pick a gate-admissible dig station that is also a legal DUMP station.

    Two properties are enforced, not hoped for:

      * the station is PERSISTENT (``fp_conservative``) whenever
        ``persistent_only`` -- legal even with the whole trench dug -- and the
        ``dist`` map it is scored against was built the same way, so the
        controller cannot drive somewhere its own excavation deletes;
      * a dump cabin with a legal ACCEPTED destination exists in the EXACT
        predicted post-dig state (``_dump_cabin_after_dig``).  An excavator
        cannot move while loaded, so a dig without that is a deadlock, not a
        setback.

    Under v2 the on-axis stations (perpendicular offset inside the retired
    3.5 m floor -- dig ahead, then retreat along the axis) are preferred by a
    score bonus; they simply do not exist under v1, where the band forbids them.

    Returns ``(key, dump_cabin, selected_cells, on_axis)`` or None.
    """
    geom = o.geom
    diggable = o.dig_targets & (action_map == 0) & (~last_dig)
    if not diggable.any():
        return None
    piles = action_map > 0
    keep_clear = dilate_linf(diggable, 6)
    cands = []
    for a in range(o.naxes):
        own = (o.membership & np.uint8(1 << a)) != 0
        fresh_a = diggable & own
        if not fresh_a.any():
            continue
        near = dilate_linf(fresh_a, reach)
        bad_a = diggable & (o.membership != 0) & (~own)
        for bh2 in range(NH):
            if not o.yaw_ok[bh2, a]:
                continue
            posemask = o.pose_ok[a, bh2] & fp_free[bh2] & near & (dist[bh2] < INF)
            if persistent_only:
                posemask = posemask & o.fp_conservative[bh2]
            poses = np.argwhere(posemask)
            if poses.shape[0] == 0:
                continue
            dd = dist[bh2][posemask]
            bonus = np.where(
                o.on_axis[a][posemask], np.int32(ON_AXIS_BONUS), np.int32(0)
            )
            # v2 drops the lateral band, so the pose-valid set for one heading
            # is every cell within reach of the section instead of a narrow
            # lane -- thousands of poses, and the twelve-cabin cone scan below
            # is linear in that.  Keep the nearest POSE_CAP (on-axis poses
            # counted as closer), which is what the score would have preferred
            # anyway; the exact-cone sweep below is the safety net.
            if poses.shape[0] > POSE_CAP:
                order = np.argpartition(dd - bonus, POSE_CAP)[:POSE_CAP]
                poses, dd, bonus = poses[order], dd[order], bonus[order]
            for cb2 in range(NH):
                offs = geom.cones[bh2][cb2]
                nsel, nbad, npad, npile = gather_counts(
                    poses, offs, [diggable, bad_a, o.padding, piles]
                )
                good = (nsel > 0) & (nbad == 0) & (npad == 0) & (npile == 0)
                if not good.any():
                    continue
                cabin_cost = min((cb2 - cb) % NH, (cb - cb2) % NH)
                idx = np.flatnonzero(good)
                score = nsel[idx] * 3 - (dd[idx] + cabin_cost) + bonus[idx]
                if idx.size > 16:
                    keep = np.argpartition(-score, 16)[:16]
                    idx, score = idx[keep], score[keep]
                for j, i in enumerate(idx):
                    key = (int(poses[i, 0]), int(poses[i, 1]), bh2, cb2)
                    if key in banned:
                        continue
                    cands.append((int(score[j]), int(nsel[i]),
                                  int(dd[i]) + cabin_cost, key))
    cands.sort(key=lambda t: (-t[0], t[2]))
    probes = 0
    for _, _nsel, cost, key in cands:
        if probes >= max_probe:
            break
        r2, c2, bh2, cb2 = key
        probes += 1
        cones12 = geom.cones_all_cabins(r2, c2, bh2)
        cone = cones12[cb2].astype(bool)
        picked = _validate_station(o, r2, c2, bh2, cb2, cones12, cone,
                                   action_map, last_dig, keep_clear)
        if picked is not None:
            dump_cb, nsel_exact = picked
            return key, dump_cb, nsel_exact, o.station_is_on_axis(r2, c2, bh2)

    # The offset-table cone is only ~81% exact, so when the fast scan finds
    # nothing, sweep the nearest reachable pose-valid poses with exact Terra
    # cones.  On-axis poses are swept first under v2.
    pool = []
    for a in range(o.naxes):
        own = (o.membership & np.uint8(1 << a)) != 0
        fresh_a = diggable & own
        if not fresh_a.any():
            continue
        near = dilate_linf(fresh_a, reach)
        for bh2 in range(NH):
            if not o.yaw_ok[bh2, a]:
                continue
            pm = o.pose_ok[a, bh2] & fp_free[bh2] & near & (dist[bh2] < INF)
            if persistent_only:
                pm = pm & o.fp_conservative[bh2]
            cells = np.argwhere(pm)
            if cells.shape[0] == 0:
                continue
            rank = dist[bh2][pm] - np.where(
                o.on_axis[a][pm], np.int32(ON_AXIS_BONUS), np.int32(0)
            )
            if cells.shape[0] > SWEEP_CAP:
                order = np.argpartition(rank, SWEEP_CAP)[:SWEEP_CAP]
                cells, rank = cells[order], rank[order]
            for (rr2, cc2), rk in zip(cells, rank):
                pool.append((int(rk), int(rr2), int(cc2), bh2))
    pool.sort()
    seen = set()
    for _rank, r2, c2, bh2 in pool[:120]:
        if (r2, c2, bh2) in seen:
            continue
        seen.add((r2, c2, bh2))
        cones12 = geom.cones_all_cabins(r2, c2, bh2)
        for cb2 in range(NH):
            key = (r2, c2, bh2, cb2)
            if key in banned:
                continue
            cone = cones12[cb2].astype(bool)
            picked = _validate_station(o, r2, c2, bh2, cb2, cones12, cone,
                                       action_map, last_dig, keep_clear)
            if picked is not None:
                dump_cb, nsel = picked
                return key, dump_cb, nsel, o.station_is_on_axis(r2, c2, bh2)
    return None


def _validate_station(o, r2, c2, bh2, cb2, cones12, cone, action_map, last_dig,
                      keep_clear):
    """Exact dig admissibility plus an exact post-dig dump. (dump_cb, nsel) or None."""
    appl, valid, nsel, _nft, sel_mask = o.dig_admissible(
        r2, c2, bh2, cone, action_map, last_dig)
    if not (appl and valid and nsel > 0):
        return None
    if np.any(o.padding[cone]):
        return None
    # The post-dig prediction assumes the clean-dirt branch of ``_apply_dig_mask``
    # (lower every selected cell by dig_depth).  Refuse the station rather than
    # predict the positive-soil branch, whose prefix apportionment would make the
    # dump-legality test approximate.
    if np.any(action_map[sel_mask] != 0):
        return None
    dump_cb = _dump_cabin_after_dig(o, r2, c2, bh2, cb2, cones12,
                                    action_map, last_dig, sel_mask, keep_clear)
    if dump_cb is None:
        return None
    return dump_cb, nsel


def attribute_stall(o: SlotOracle, action_map, last_dig, fp_free, dist):
    """Why can no admissible dig station be found for the remaining cells?

    Applies the filters cumulatively over every (pose, cabin) that could dig a
    remaining cell and reports, per remaining target cell, the first filter that
    removes every option.  Uses the offset-table cone (mean 0.35 cell error) for
    the sweep, so single-cell attributions are indicative rather than exact.
    """
    geom = o.geom
    diggable = o.dig_targets & (action_map == 0) & (~last_dig)
    remaining = np.argwhere(diggable)
    if remaining.shape[0] == 0:
        return {"remaining_cells": 0}
    n = remaining.shape[0]
    index = -np.ones(SHAPE, dtype=np.int32)
    index[remaining[:, 0], remaining[:, 1]] = np.arange(n)
    pose_stage = ("pose_in_band_and_parallel" if o.standoff_enforced
                  else "pose_yaw_parallel")
    stages = ["in_cone_without_obstacle", pose_stage,
              "no_perpendicular_veto", "no_pile_in_cone",
              "pose_footprint_legal", "pose_persistent", "pose_reachable",
              "legal_dump_after_dig"]
    reach = {k: np.zeros(n, dtype=bool) for k in stages}
    piles = action_map > 0
    dump_cache = {}
    cone_cache = {}
    # The dump stage costs one exact Terra cone bundle plus a twelve-cabin dump
    # scan per candidate pose.  Under v1 the lateral band kept that population
    # tiny; v2 admits every yaw-parallel pose in reach, so it has to be bounded
    # or the attribution costs more than the episode it explains.  Poses past
    # the budget are simply not credited with the last stage, which makes the
    # final number a LOWER bound on "has a legal dump", never an over-claim.
    dump_budget = 400
    dump_budget_hit = False

    def cones_for(pr, pc, bh):
        key = (pr, pc, bh)
        if key not in cone_cache:
            cone_cache[key] = geom.cones_all_cabins(pr, pc, bh)
        return cone_cache[key]

    for bh2 in range(NH):
        poses = np.argwhere(dilate_linf(diggable, 12))
        if poses.shape[0] == 0:
            continue
        pbits = np.array([o.pose_bits(int(pr), int(pc), bh2) for pr, pc in poses],
                         dtype=np.uint8)
        fpok = fp_free[bh2][poses[:, 0], poses[:, 1]]
        persistent = o.fp_conservative[bh2][poses[:, 0], poses[:, 1]]
        reachable = dist[bh2][poses[:, 0], poses[:, 1]] < INF
        for cb2 in range(NH):
            offs = geom.cones[bh2][cb2]
            rr = poses[:, 0:1] + offs[None, :, 0]
            cc = poses[:, 1:2] + offs[None, :, 1]
            ok = (rr >= 0) & (rr < SHAPE[0]) & (cc >= 0) & (cc < SHAPE[1])
            rrc, ccc = np.clip(rr, 0, SHAPE[0] - 1), np.clip(cc, 0, SHAPE[1] - 1)
            freshm = diggable[rrc, ccc] & ok
            live = freshm.any(1)
            nopad = ~np.any(o.padding[rrc, ccc] & ok, axis=1)
            nopile = ~np.any(piles[rrc, ccc] & ok, axis=1)
            mem = o.membership[rrc, ccc]
            for i in np.flatnonzero(live & nopad):
                cells = index[rrc[i][freshm[i]], ccc[i][freshm[i]]]
                cells = cells[cells >= 0]
                reach["in_cone_without_obstacle"][cells] = True
                b = pbits[i]
                if b == 0:
                    continue
                reach[pose_stage][cells] = True
                m = mem[i][freshm[i]]
                tr = m != 0
                if tr.any() and not ((m[tr] & b) != 0).all():
                    continue
                reach["no_perpendicular_veto"][cells] = True
                if not nopile[i]:
                    continue
                reach["no_pile_in_cone"][cells] = True
                if not fpok[i]:
                    continue
                reach["pose_footprint_legal"][cells] = True
                if not persistent[i]:
                    continue
                reach["pose_persistent"][cells] = True
                if not reachable[i]:
                    continue
                reach["pose_reachable"][cells] = True
                pr, pc = int(poses[i, 0]), int(poses[i, 1])
                ck = (pr, pc, bh2, cb2)
                if ck not in dump_cache:
                    if len(dump_cache) >= dump_budget:
                        dump_budget_hit = True
                        continue
                    cones12 = cones_for(pr, pc, bh2)
                    cone_exact = cones12[cb2].astype(bool)
                    dump_cache[ck] = _validate_station(
                        o, pr, pc, bh2, cb2, cones12, cone_exact,
                        action_map, last_dig, None)
                if dump_cache[ck] is None:
                    continue
                reach["legal_dump_after_dig"][cells] = True
    out = {"remaining_cells": int(n)}
    for k in stages:
        out[k] = int(reach[k].sum())
    prev = np.ones(n, dtype=bool)
    binding = {}
    for k in stages:
        lost = int(np.sum(prev & ~reach[k]))
        if lost:
            binding[k] = lost
        prev = prev & reach[k]
    out["binding_stage_cells"] = binding
    out["cells_with_no_option_at_all"] = int(np.sum(~reach[stages[-1]]))
    out["dump_stage_budget_exhausted"] = bool(dump_budget_hit)
    out["dump_stage_poses_tested"] = len(dump_cache)
    return out


def make_terra_probe():
    """Terra-side ground truth for the gate, vmapped over slots.

    Verbatim transcription of the ``applicable`` predicate in
    ``State._get_fresh_trench_dig_alignment_details`` plus the counts needed to
    localise any disagreement with the numpy replica.
    """
    from terra.state import _as_2d_map, _as_axes_table, _as_scalar_int

    def probe(state):
        cur = state._get_current_agent_state()
        target = _as_2d_map(state.world.target_map.map)
        action = _as_2d_map(state.world.action_map.map)
        records = _as_axes_table(state.world.trench_axes).astype(jnp.float32)
        max_axes = records.shape[0]
        trench_type = jnp.clip(_as_scalar_int(state.world.trench_type), 0, max_axes)
        raw_cone = state._build_dig_dump_cone()
        dig_mask = state._mask_out_wrong_dig_tiles(raw_cone)
        dig_mask = jnp.asarray(dig_mask, dtype=jnp.bool_).reshape(-1)
        dig_2d = dig_mask.reshape(target.shape)
        fresh = jnp.logical_and(dig_2d, jnp.logical_and(target < 0, action == 0))
        fresh_trench = jnp.logical_and(
            fresh, state.world.trench_axis_membership != jnp.uint8(0))
        applicable = jnp.logical_and(
            cur.agent_type[0] == 0,
            jnp.logical_and(
                cur.loaded[0] == 0,
                jnp.logical_and(trench_type > 0, jnp.any(fresh_trench))))
        valid, _, _ = state._get_fresh_trench_dig_alignment()
        return (applicable, valid,
                jnp.sum(fresh_trench.astype(jnp.int32)),
                jnp.sum(jnp.asarray(raw_cone, dtype=jnp.int32)),
                jnp.sum(dig_mask.astype(jnp.int32)),
                jnp.asarray(raw_cone, dtype=jnp.bool_).reshape(target.shape),
                cur.pos_base[0].astype(jnp.int32),
                cur.pos_base[1].astype(jnp.int32),
                cur.angle_base[0].astype(jnp.int32),
                cur.angle_cabin[0].astype(jnp.int32),
                state.agent.current_agent.astype(jnp.int32))

    return jax.jit(jax.vmap(probe))


def select_slots(rows, include_families, include_cells, exclude_cells):
    out = []
    for row in rows:
        if include_families and row["family"] not in include_families:
            continue
        cell = row["primary_cell"]
        if include_cells and not any(fnmatch.fnmatch(cell, p) for p in include_cells):
            continue
        if exclude_cells and any(fnmatch.fnmatch(cell, p) for p in exclude_cells):
            continue
        out.append(int(row["slot_index"]) - 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank-root", type=Path, required=True)
    ap.add_argument("--accepted-panel", default="development")
    ap.add_argument("--panel-family", default="gate_main")
    ap.add_argument("--terra-revision", required=True)
    ap.add_argument("--include-family", nargs="*", default=["trench"])
    ap.add_argument("--include-cell", nargs="*", default=[])
    ap.add_argument("--exclude-cell", nargs="*", default=["trn-net4-*"])
    ap.add_argument(
        "--include-net4", action="store_true",
        help="also run the 48 trn-net4-* slots (clears the default exclusion)")
    ap.add_argument(
        "--gate-v1", action="store_true",
        help="run the retired v1 gate semantics (perpendicular standoff band "
             "ENFORCED, EnvConfig.trench_dig_standoff_enforced=True) instead of "
             "the shipped v2 yaw-parallel-only semantics")
    ap.add_argument("--slot-limit-per-cell", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=450)
    ap.add_argument("--extended-horizon", type=int, default=0)
    ap.add_argument("--verify-action-mask", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bank_root = args.bank_root.resolve()
    exclude_cells = [] if args.include_net4 else list(args.exclude_cell)

    release_id = json.loads((bank_root / "dataset.json").read_text()).get("release_id")
    accepted_bank = load_accepted_bank(
        bank_root, "G-UNIFORM", args.terra_revision,
        curriculum_stage="full" if release_id == V8_RELEASE_ID else None,
        evaluation_panel_family=args.panel_family,
    )
    panel = next(p for p in accepted_bank.evaluation_panels if p.name == args.accepted_panel)
    rows = load_manifest(bank_root / panel.maps_path)
    panel_count = len(rows)

    slot_indices = select_slots(
        rows, set(args.include_family) if args.include_family else set(),
        list(args.include_cell), exclude_cells,
    )
    if args.slot_limit_per_cell > 0:
        seen, kept = {}, []
        for si in slot_indices:
            cell = rows[si]["primary_cell"]
            seen[cell] = seen.get(cell, 0) + 1
            if seen[cell] <= args.slot_limit_per_cell:
                kept.append(si)
        slot_indices = kept
    if not slot_indices:
        raise ValueError("slot selection is empty")
    count = len(slot_indices)
    slot_rows = [rows[i] for i in slot_indices]
    print(f"slots={count} conditions={len(set(r['primary_cell'] for r in slot_rows))}",
          flush=True)

    checkpoint = load_pkl_object(str(args.checkpoint))
    os.environ["DATASET_PATH"] = str(bank_root)
    os.environ["DATASET_SIZE"] = str(panel_count)
    config = configure_for_bank(checkpoint["train_config"], panel.maps_path, count)
    if not bool(getattr(config, "enforce_trench_dig_alignment", False)):
        raise RuntimeError("this oracle must run with the gate ON")
    # The gate semantics are an explicit CLI choice, never inherited silently:
    # v1 and v2 admit different stations, so a run that does not say which one
    # it measured is not comparable with anything.
    if not hasattr(config, "trench_dig_standoff_enforced"):
        raise RuntimeError(
            "this baselines revision predates the gate-semantics selector; "
            "trench_dig_standoff_enforced is required")
    config.trench_dig_standoff_enforced = bool(args.gate_v1)
    print(f"gate semantics: {'v1 (standoff band ENFORCED)' if args.gate_v1 else 'v2 (yaw-parallel only)'}",
          flush=True)
    _, env, env_params, _ = make_mixed_agent_states(config)
    env_params = jax.tree_util.tree_map(lambda v: v[0], env_params)

    all_map_keys = np.asarray(exact_reset_keys(panel_count))
    all_state_keys = np.asarray(
        manifest_environment_keys(rows, panel_count,
                                  accepted_bank.environment_protocol_sha256)
    )
    timestep, env_params, state_keys = prepare_manifest_episode_reset(
        env, env_params,
        jnp.asarray(all_map_keys[np.asarray(slot_indices)]),
        jnp.asarray(all_state_keys[np.asarray(slot_indices)]),
    )
    if not bool(np.ravel(np.asarray(timestep.env_cfg.enforce_trench_dig_alignment))[0]):
        raise RuntimeError("resolved env gate flag is off")

    resolved = timestep.env_cfg
    tile_size = float(np.ravel(np.asarray(resolved.tile_size))[0])
    yaw_tol = float(np.ravel(np.asarray(resolved.trench_dig_yaw_tolerance_rad))[0])
    so_min = float(np.ravel(np.asarray(resolved.trench_dig_standoff_min_m))[0])
    so_max = float(np.ravel(np.asarray(resolved.trench_dig_standoff_max_m))[0])
    standoff_enforced = bool(
        np.ravel(np.asarray(resolved.trench_dig_standoff_enforced))[0]
    )
    if standoff_enforced != bool(args.gate_v1):
        raise RuntimeError(
            f"resolved env gate semantics standoff_enforced={standoff_enforced} "
            f"but --gate-v1={bool(args.gate_v1)} was requested")
    dig_depth = int(np.ravel(np.asarray(resolved.agent.dig_depth))[0])
    agent_w = int(np.ravel(np.asarray(resolved.agent.width))[0])
    agent_h = int(np.ravel(np.asarray(resolved.agent.height))[0])
    if tile_size <= 0 or agent_w <= 0 or agent_h <= 0:
        raise RuntimeError("resolved env config has degenerate geometry")

    # Derive the geometry config from the RESOLVED env config leaf by leaf, so no
    # field can silently fall back to an EnvConfig() default that the trained
    # arm overrode (a hand-assembled config produced a cone off by up to 5 cells).
    def _scalarize(x):
        arr = np.asarray(x)
        if arr.ndim == 0:
            return arr.item()
        return np.ravel(arr)[0].item()

    geo_cfg = jax.tree_util.tree_map(_scalarize, resolved)
    if int(geo_cfg.agent.width) != agent_w or int(geo_cfg.agent.height) != agent_h:
        raise RuntimeError("scalarized env config lost the agent dimensions")
    t0 = time.time()
    geom = Geometry(geo_cfg)
    geom_report = geom.verify_translation()
    print(f"geometry {time.time() - t0:.1f}s tile={tile_size:.6f} agent={agent_w}x{agent_h} "
          f"fwd0={geom.fwd[0]} bwd0={geom.bwd[0]} {geom_report}", flush=True)

    st = timestep.state
    target = np.asarray(st.world.target_map.map).reshape(count, *SHAPE)
    padding = np.asarray(st.world.padding_mask.map).reshape(count, *SHAPE)
    dump_init = np.asarray(st.world.dumpability_mask_init.map).reshape(count, *SHAPE)
    records = np.asarray(st.world.trench_axes).astype(np.float32).reshape(count, 4, 8)
    ttype = np.asarray(st.world.trench_type).reshape(-1)
    membership = np.asarray(st.world.trench_axis_membership).reshape(count, *SHAPE).astype(np.uint8)
    static_base = np.asarray(st.world.static_traversability_base.map).reshape(count, *SHAPE)
    if np.any(np.asarray(st.world.action_map.map) != 0):
        raise RuntimeError("panel reset is not a full-start reset")

    oracles = []
    for k in range(count):
        if not np.array_equal(static_base[k].astype(bool), padding[k].astype(bool)):
            raise RuntimeError(f"slot {k}: static traversability base != padding mask")
        oracles.append(SlotOracle(
            k, slot_rows[k]["primary_cell"], target[k], padding[k], dump_init[k],
            records[k], ttype[k], membership[k], geom, tile_size, yaw_tol, so_min, so_max,
            standoff_enforced, dig_depth,
        ))
    required = np.array([int((target[k] < 0).sum()) for k in range(count)])

    terra_probe = make_terra_probe()
    mask_fn = None
    if args.verify_action_mask:
        mask_fn = jax.jit(jax.vmap(lambda s: s._get_action_mask_tracked()))

    horizon = args.extended_horizon or args.horizon
    terminated = np.zeros(count, dtype=bool)
    succeeded = np.zeros(count, dtype=bool)
    success_step = np.full(count, -1, dtype=np.int32)
    ep_len = np.zeros(count, dtype=np.int32)
    banned_station = [set() for _ in range(count)]
    banned_dump = [set() for _ in range(count)]
    action_counts = np.zeros((count, 8), dtype=np.int32)
    no_effect = np.zeros(count, dtype=np.int32)
    align_checks = 0
    mask_checks = 0
    cone_diff_cells = 0
    cone_diff_steps = 0
    cone_compared = 0
    accepted_stack = np.stack([o.accepted for o in oracles])
    illegal_by_action = np.zeros(8, dtype=np.int64)
    illegal_events = []
    prev_illegal = np.zeros(count, dtype=np.int64)
    fp_cache = [None] * count
    dist_cache = [None] * count

    t_start = time.time()
    # Coarse wall-clock accounting, printed with the progress line.  The v2
    # station search scans a much larger pose set than v1 (no lateral band), so
    # knowing where the time goes is the difference between "slow" and "wrong".
    timers = {"terra_step": 0.0, "probe": 0.0, "plan": 0.0, "bfs": 0.0,
              "stall_attribution": 0.0}
    for step in range(horizon):
        active = ~terminated
        pos = np.asarray(st.agent.agent_states[0].pos_base).reshape(count, 2).astype(np.int32)
        bh_all = np.asarray(st.agent.agent_states[0].angle_base).reshape(-1).astype(np.int32)
        cb_all = np.asarray(st.agent.agent_states[0].angle_cabin).reshape(-1).astype(np.int32)
        loaded_all = np.asarray(st.agent.agent_states[0].loaded).reshape(-1).astype(np.int32)
        action_maps = np.asarray(st.world.action_map.map).reshape(count, *SHAPE).astype(np.int32)
        last_digs = np.asarray(st.world.last_dig_mask.map).reshape(count, *SHAPE).astype(bool)
        obs_valid = np.asarray(
            timestep.observation["fresh_trench_dig_alignment_valid"]
        ).reshape(-1)
        cur_cones = geom.cones_batch(pos[:, 0], pos[:, 1], bh_all, cb_all)
        _t = time.time()
        _pr = terra_probe(st)
        t_cone_mask = _pr[5]
        (t_appl, t_valid, t_nfresh, t_ncone, t_nsel) = [
            np.asarray(v).reshape(-1) for v in _pr[:5]]
        (t_row, t_col, t_bh, t_cb, t_cur) = [
            np.asarray(v).reshape(-1) for v in _pr[6:]]
        bad = (t_row != pos[:, 0]) | (t_col != pos[:, 1]) | (t_bh != bh_all) | (t_cb != cb_all)
        if bad.any():
            k0 = int(np.flatnonzero(bad)[0])
            raise RuntimeError(
                f"step {step} slot {k0}: pose read from agent_states[0] "
                f"{(int(pos[k0,0]), int(pos[k0,1]), int(bh_all[k0]), int(cb_all[k0]))} "
                f"!= Terra's current agent "
                f"{(int(t_row[k0]), int(t_col[k0]), int(t_bh[k0]), int(t_cb[k0]))} "
                f"current_agent={int(t_cur[k0])}")
        # The synthetic-state cone agrees with Terra's live cone except on cells
        # that sit exactly on the +/-30 deg wedge boundary (float32 arctan2
        # rounding).  Report the size of that disagreement and use Terra's own
        # cone for every decision, so the oracle never acts on the approximation.
        syn = np.asarray([int(cur_cones[i].sum()) for i in range(count)])
        cone_diff_cells += int(np.abs(syn - t_ncone).sum())
        cone_diff_steps += int(np.sum(syn != t_ncone))
        cone_compared += count
        cur_cones = np.asarray(t_cone_mask)
        timers["probe"] += time.time() - _t
        if False:
            raise RuntimeError(
                "unreachable")
        terra_mask = None
        if mask_fn is not None:
            terra_mask = np.asarray(mask_fn(st)).reshape(count, -1).astype(bool)

        actions = np.full(count, DO_NOTHING, dtype=np.int32)
        for k in range(count):
            if not active[k]:
                continue
            o = oracles[k]
            r, c, bh, cb = int(pos[k, 0]), int(pos[k, 1]), int(bh_all[k]), int(cb_all[k])
            am, ld = action_maps[k], last_digs[k]

            appl, val, nsel_r, nft, _selm = o.dig_admissible(
                r, c, bh, cur_cones[k].astype(bool), am, ld
            )
            # Terra's applicability also requires an empty excavator.
            appl = appl and int(loaded_all[k]) == 0
            replica_export = val if appl else True
            if bool(obs_valid[k] > 0.5) != bool(replica_export):
                raise RuntimeError(
                    f"step {step} slot {k}: replica alignment {replica_export} != "
                    f"Terra export {obs_valid[k]} (pose r={r} c={c} bh={bh} cb={cb} "
                    f"loaded={int(loaded_all[k])} bits={o.pose_bits(r, c, bh)} | "
                    f"replica appl={appl} fresh_trench={nft} sel={nsel_r} "
                    f"cone={int(cur_cones[k].sum())} | "
                    f"terra appl={bool(t_appl[k])} valid={bool(t_valid[k])} "
                    f"fresh_trench={int(t_nfresh[k])} cone={int(t_ncone[k])} "
                    f"sel={int(t_nsel[k])} | standoff="
                    f"{[round(float(o.standoff[a, r, c]), 4) for a in range(o.naxes)]} "
                    f"yaw={[round(float(o.yaw_err[bh, a]), 4) for a in range(o.naxes)]})"
                )
            align_checks += 1

            if fp_cache[k] is None:
                fp_cache[k] = footprint_free(blocked_mask(am, o.padding), geom)
            fp_free = fp_cache[k]

            if terra_mask is not None:
                pred = np.zeros(6, dtype=bool)
                if int(loaded_all[k]) == 0:
                    flat_free = fp_free.reshape(NH, -1)
                    here = r * SHAPE[1] + c
                    for a in range(2):
                        d = int(geom.succ_flat[bh, a][here])
                        pred[a] = d >= 0 and bool(flat_free[bh, d]) and d != here
                    pred[2] = bool(fp_free[(bh - 1) % NH, r, c])
                    pred[3] = bool(fp_free[(bh + 1) % NH, r, c])
                pred[4] = True
                pred[5] = True
                if not np.array_equal(pred, terra_mask[k][:6]):
                    raise RuntimeError(
                        f"step {step} slot {k}: replica move mask {pred.astype(int)} != "
                        f"Terra {terra_mask[k][:6].astype(int)} loaded={int(loaded_all[k])}"
                    )
                mask_checks += 1

            if int(loaded_all[k]) > 0:
                if o.phase != o.DUMP or not o.plan:
                    occ = o.occupied_mask(r, c, bh)
                    cones12 = geom.cones_all_cabins(r, c, bh)
                    ctx = o.dump_context(am, occ)
                    keep_clear = dilate_linf(
                        o.dig_targets & (am == 0) & (~ld), 6)
                    ranked = []
                    for cbd in range(NH):
                        if (r, c, bh, cbd) in banned_dump[k]:
                            continue
                        cone_d = cones12[cbd].astype(bool)
                        legal = o.dump_legal_mask(cone_d, am, ld, occ, ctx)
                        n = int(legal.sum())
                        if n == 0:
                            continue
                        far = not bool(np.any(legal & keep_clear))
                        interior = not bool(np.any(legal & ~o.accepted_interior))
                        ranked.append((int(far) * 2 + int(interior), n, cbd))
                    if not ranked:
                        # Deadlock: an excavator cannot move while loaded, so
                        # only a cabin rotation could help and none does.  The
                        # controller waits rather than dumping into the
                        # fallback branch, which would drop the load outside
                        # target > 0 and cap dump_purity for the episode.
                        o.stuck_loaded += 1
                        o.note("loaded_no_legal_dump")
                        continue
                    ranked.sort(key=lambda t: (-t[0], -t[1]))
                    best_cb = ranked[0][2]
                    o.plan = cabin_actions(cb, best_cb) + [DO]
                    o.dump_cabin = best_cb
                    o.phase = o.DUMP
                # Re-validate the dump against Terra's own live cone in the very
                # step that executes it.  A dump with no legal accepted cell
                # falls back to non-accepted ground and is unrecoverable spoil,
                # so it is never pressed.
                if o.plan and o.plan[0] == DO:
                    occ_now = o.occupied_mask(r, c, bh)
                    if o.dump_legal_count(
                        cur_cones[k].astype(bool), am, ld, occ_now
                    ) == 0:
                        banned_dump[k].add((r, c, bh, cb))
                        o.plan = []
                        o.phase = o.IDLE
                        o.dump_precheck_blocks += 1
                        o.note("dump_precheck_failed")
                        continue
            else:
                if o.phase == o.DUMP:
                    o.phase = o.IDLE
                    o.plan = []
                    banned_dump[k].clear()
                if not o.plan and o.blocked_until_change:
                    o.no_candidate += 1
                    continue
                if not o.plan:
                    # Navigate over PERSISTENT poses only (legal even with the
                    # whole trench dug), with a first-leg exception out of the
                    # spawn pocket.  This is what stops the controller walling
                    # itself off behind its own excavation.
                    _t = time.time()
                    dist = pose_bfs((r, c, bh), fp_free, geom,
                                    persist=o.fp_conservative)
                    dist_cache[k] = dist
                    timers["bfs"] += time.time() - _t
                    _t = time.time()
                    picked = choose_station(o, r, c, bh, cb, am, ld,
                                           banned_station[k], fp_free, dist)
                    timers["plan"] += time.time() - _t
                    o.replans += 1
                    if picked is None:
                        o.no_candidate += 1
                        o.blocked_until_change = True
                        o.note("no_admissible_station")
                        if o.stall_attribution is None:
                            _t = time.time()
                            o.stall_attribution = attribute_stall(
                                o, am, ld, fp_free, dist)
                            timers["stall_attribution"] += time.time() - _t
                            o.stall_attribution["step"] = step
                        continue
                    (sr, sc, sbh, scb), dump_cb, _n, on_axis = picked
                    nav = reconstruct_actions(dist, (sr, sc, sbh), geom,
                                              persist=o.fp_conservative)
                    if nav is None:
                        banned_station[k].add((sr, sc, sbh, scb))
                        o.note("unreachable_station")
                        continue
                    o.plan = nav + cabin_actions(cb, scb) + [DO]
                    o.goal = (sr, sc, sbh, scb)
                    o.dump_cabin = dump_cb
                    o.phase = o.NAV
                    o.stations += 1
                    o.stations_on_axis += int(bool(on_axis))
                # re-validate a dig right before executing it
                if o.plan and o.plan[0] == DO:
                    appl2, val2, nsel2, _nft2, _sm2 = o.dig_admissible(
                        r, c, bh, cur_cones[k].astype(bool), am, ld
                    )
                    if not (appl2 and val2 and nsel2 > 0):
                        if o.goal is not None:
                            banned_station[k].add(o.goal)
                        o.plan = []
                        o.note("dig_precheck_failed")
                        continue
            actions[k] = o.plan.pop(0)

        prev_pos = pos.copy()
        prev_bh = bh_all.copy()
        prev_loaded = loaded_all.copy()

        _t = time.time()
        wrapped = wrap_action(jnp.asarray(actions, dtype=jnp.int32), env.batch_cfg.action_type)
        candidate = env.step_no_reset(
            timestep, wrapped, jax.random.split(jax.random.PRNGKey(step), count)
        )
        jax.block_until_ready(candidate.reward)
        timers["terra_step"] += time.time() - _t
        act_j = jnp.asarray(active)

        def _preserve(previous, cand):
            if not hasattr(cand, "shape"):
                return cand
            if cand.ndim == 0 or cand.shape[0] != count:
                return cand
            mask = act_j.reshape((count,) + (1,) * (cand.ndim - 1))
            return jnp.where(mask, cand, previous)

        timestep = jax.tree_util.tree_map(_preserve, timestep, candidate)
        st = timestep.state

        new_pos = np.asarray(st.agent.agent_states[0].pos_base).reshape(count, 2).astype(np.int32)
        new_bh = np.asarray(st.agent.agent_states[0].angle_base).reshape(-1).astype(np.int32)
        new_loaded = np.asarray(st.agent.agent_states[0].loaded).reshape(-1).astype(np.int32)
        had_effect = np.asarray(timestep.info["action_had_effect"]).reshape(-1).astype(bool)

        # positive soil sitting outside the accepted dump zone, per slot.  Any
        # increase is attributed to the action just executed, which separates
        # dig-side soil relaxation from the dump-mask fallback.
        post = np.asarray(st.world.action_map.map).reshape(count, *SHAPE).astype(np.int64)
        pos = np.clip(post, 0, None)
        illegal_now = np.sum(np.where(accepted_stack, 0, pos), axis=(1, 2))
        delta_illegal = illegal_now - prev_illegal
        for k in np.flatnonzero((delta_illegal != 0) & active):
            a = int(actions[k])
            illegal_by_action[a] += int(delta_illegal[k])
            if len(illegal_events) < 400:
                illegal_events.append({
                    "step": step, "slot": int(k), "condition": oracles[k].cell,
                    "action": ACTION_NAMES[a], "delta": int(delta_illegal[k]),
                    "loaded_before": int(prev_loaded[k]),
                    "loaded_after": int(new_loaded[k]),
                })
        prev_illegal = illegal_now

        for k in range(count):
            if not active[k]:
                continue
            o = oracles[k]
            a = int(actions[k])
            action_counts[k, a] += 1
            if not had_effect[k]:
                no_effect[k] += 1
            if a in (FORWARD, BACKWARD, CLOCK, ANTICLOCK):
                moved = (
                    new_pos[k, 0] != prev_pos[k, 0] or new_pos[k, 1] != prev_pos[k, 1]
                    or new_bh[k] != prev_bh[k]
                )
                if not moved:
                    o.plan = []
                    o.phase = o.IDLE
                    o.move_refused += 1
                    o.note("move_refused")
                    fp_cache[k] = None
                else:
                    o.blocked_until_change = False
            elif a == DO:
                fp_cache[k] = None
                o.blocked_until_change = False
                if prev_loaded[k] == 0:
                    if new_loaded[k] > 0:
                        o.digs += 1
                    else:
                        o.failed_digs += 1
                        if o.goal is not None:
                            banned_station[k].add(o.goal)
                        o.plan = []
                        o.phase = o.IDLE
                        o.note("dig_refused")
                else:
                    if new_loaded[k] == 0:
                        o.dumps += 1
                        o.plan = []
                        o.phase = o.IDLE
                        banned_dump[k].clear()
                    else:
                        o.failed_dumps += 1
                        banned_dump[k].add((int(prev_pos[k, 0]), int(prev_pos[k, 1]),
                                            int(prev_bh[k]), int(o.dump_cabin)))
                        o.plan = []
                        o.note("dump_refused")

        ep_len += active.astype(np.int32)
        step_done = np.asarray(timestep.done).reshape(-1).astype(bool)
        step_succ = np.asarray(timestep.info["task_done"]).reshape(-1).astype(bool)
        newly = active & step_succ & ~succeeded
        success_step[newly] = step + 1
        succeeded |= active & step_succ
        # With --extended-horizon we want the true step budget, so ignore Terra's
        # own 450-step timeout and stop a slot only when the task is complete.
        terminated |= active & (step_succ if args.extended_horizon else step_done)

        if step % 10 == 0 or terminated.all():
            spent = " ".join(f"{k}={v:.0f}s" for k, v in timers.items() if v >= 1.0)
            print(f"step {step + 1} done={int(terminated.sum())}/{count} "
                  f"succ={int(succeeded.sum())} elapsed={time.time() - t_start:.0f}s "
                  f"[{spent}]",
                  flush=True)
        if terminated.all():
            break

    final = np.asarray(st.world.action_map.map).reshape(count, *SHAPE).astype(np.int32)
    dug = np.array([int(np.sum((final[k] < 0) & (target[k] < 0))) for k in range(count)])
    dig_fraction = dug / np.maximum(required, 1)

    per_slot = []
    for k in range(count):
        o = oracles[k]
        per_slot.append({
            "batch_index": k,
            "slot_index": slot_indices[k] + 1,
            "condition": o.cell,
            "map_id": slot_rows[k].get("map_id", ""),
            "axes": o.naxes,
            "succeeded": bool(succeeded[k]),
            "success_step": int(success_step[k]),
            "within_horizon": bool(succeeded[k] and 0 < success_step[k] <= args.horizon),
            "episode_length": int(ep_len[k]),
            "required_cells": int(required[k]),
            "dug_cells": int(dug[k]),
            "dig_fraction": float(dig_fraction[k]),
            "positive_soil": int(np.sum(np.clip(final[k], 0, None))),
            "accepted_soil": int(np.sum(np.where(o.accepted, np.clip(final[k], 0, None), 0))),
            "illegal_soil": int(
                np.sum(np.clip(final[k], 0, None))
                - np.sum(np.where(o.accepted, np.clip(final[k], 0, None), 0))
            ),
            "stations": o.stations, "stations_on_axis": o.stations_on_axis,
            "digs": o.digs, "dumps": o.dumps,
            "failed_digs": o.failed_digs, "failed_dumps": o.failed_dumps,
            "replans": o.replans, "no_candidate_steps": o.no_candidate,
            "stuck_loaded_steps": o.stuck_loaded, "move_refused": o.move_refused,
            "dump_precheck_blocks": o.dump_precheck_blocks,
            # An excavator cannot move while loaded, so "ended loaded with no
            # legal accepted dump from any cabin" is a hard deadlock, not a
            # slow episode.
            "loaded_deadlock": bool(
                (not succeeded[k]) and o.stuck_loaded > 0
                and int(np.ravel(np.asarray(
                    st.agent.agent_states[0].loaded))[k]) > 0
            ),
            "no_effect_actions": int(no_effect[k]),
            "reasons": o.reasons,
            "stall_attribution": o.stall_attribution,
            "action_counts": action_counts[k].tolist(),
        })

    by_cond = {}
    for row in per_slot:
        e = by_cond.setdefault(
            row["condition"],
            {"slots": 0, "within": 0, "succ": 0, "steps": [], "all_steps": [],
             "df": [], "deadlock": 0, "spoil": 0},
        )
        e["slots"] += 1
        e["within"] += int(row["within_horizon"])
        e["succ"] += int(row["succeeded"])
        if row["within_horizon"]:
            e["steps"].append(row["success_step"])
        if row["succeeded"]:
            e["all_steps"].append(row["success_step"])
        e["df"].append(row["dig_fraction"])
        e["deadlock"] += int(row["loaded_deadlock"])
        e["spoil"] += int(row["illegal_soil"])
    summary_by_condition = []
    for cond in sorted(by_cond):
        e = by_cond[cond]
        summary_by_condition.append({
            "condition": cond, "slots": e["slots"], "within_horizon": e["within"],
            "rate": e["within"] / e["slots"],
            "succeeded_any_horizon": e["succ"],
            "median_success_step": float(np.median(e["steps"])) if e["steps"] else None,
            "p90_success_step": (
                float(np.percentile(e["steps"], 90)) if e["steps"] else None),
            "max_success_step": int(max(e["steps"])) if e["steps"] else None,
            "median_success_step_any_horizon": (
                float(np.median(e["all_steps"])) if e["all_steps"] else None),
            "max_success_step_any_horizon": (
                int(max(e["all_steps"])) if e["all_steps"] else None),
            "mean_dig_fraction": float(np.mean(e["df"])),
            "loaded_deadlocks": e["deadlock"],
            "illegal_soil_units": e["spoil"],
        })

    steps_ok = [r["success_step"] for r in per_slot if r["within_horizon"]]
    steps_all = [r["success_step"] for r in per_slot if r["succeeded"]]
    payload = {
        "schema": SCHEMA,
        "contract": {
            "bank": str(bank_root),
            "panel": f"{args.panel_family}/{args.accepted_panel}",
            "terra_revision": args.terra_revision,
            "checkpoint": str(args.checkpoint),
            "gate": True, "horizon": args.horizon, "stepped_horizon": horizon,
            "gate_semantics": "v1" if args.gate_v1 else "v2",
            "trench_dig_standoff_enforced": standoff_enforced,
            "yaw_tolerance_rad": yaw_tol,
            "standoff_band_m": [so_min, so_max],
            "navigation": "persistent_poses_with_spawn_first_leg_exception",
            "slots": count,
            "verify_action_mask": bool(args.verify_action_mask),
            "alignment_export_checks": align_checks,
            "action_mask_checks": mask_checks,
            "synthetic_cone_vs_terra_compared": cone_compared,
            "synthetic_cone_vs_terra_differing_poses": cone_diff_steps,
            "synthetic_cone_vs_terra_total_cell_diff": cone_diff_cells,
            "illegal_soil_by_action": {
                ACTION_NAMES[i]: int(illegal_by_action[i]) for i in range(8)
                if illegal_by_action[i] != 0
            },
            "illegal_soil_events": illegal_events,
            "geometry": geom_report,
            "wall_seconds": time.time() - t_start,
            "wall_seconds_by_phase": {k: round(v, 1) for k, v in timers.items()},
        },
        "summary": {
            "slots": count,
            "succeeded": int(succeeded.sum()),
            "within_horizon": int(sum(r["within_horizon"] for r in per_slot)),
            "rate": float(sum(r["within_horizon"] for r in per_slot) / count),
            "median_success_step": float(np.median(steps_ok)) if steps_ok else None,
            "p90_success_step": float(np.percentile(steps_ok, 90)) if steps_ok else None,
            "max_success_step": int(max(steps_ok)) if steps_ok else None,
            "median_success_step_any_horizon": (
                float(np.median(steps_all)) if steps_all else None),
            "p90_success_step_any_horizon": (
                float(np.percentile(steps_all, 90)) if steps_all else None),
            "max_success_step_any_horizon": (
                int(max(steps_all)) if steps_all else None),
            "mean_dig_fraction": float(np.mean(dig_fraction)),
            "stations": int(sum(r["stations"] for r in per_slot)),
            "stations_on_axis": int(sum(r["stations_on_axis"] for r in per_slot)),
            "loaded_no_legal_dump_deadlocks": int(
                sum(r["loaded_deadlock"] for r in per_slot)),
            "slots_with_stuck_loaded_steps": int(
                sum(1 for r in per_slot if r["stuck_loaded_steps"] > 0)),
            "dump_precheck_blocks": int(
                sum(r["dump_precheck_blocks"] for r in per_slot)),
            "illegal_soil_units": int(sum(r["illegal_soil"] for r in per_slot)),
            "slots_with_illegal_soil": int(
                sum(1 for r in per_slot if r["illegal_soil"] > 0)),
        },
        "summary_by_condition": summary_by_condition,
        "per_slot": per_slot,
    }
    output.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["summary"], indent=1))
    for row in summary_by_condition:
        print(f"{row['condition']:26s} {row['within_horizon']:2d}/{row['slots']:2d} "
              f"(any-horizon {row['succeeded_any_horizon']:2d}) "
              f"median={row['median_success_step']} p90={row['p90_success_step']} "
              f"max={row['max_success_step']} "
              f"digfrac={row['mean_dig_fraction']:.3f} "
              f"deadlock={row['loaded_deadlocks']} spoil={row['illegal_soil_units']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
