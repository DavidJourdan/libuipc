# CLAUDE.md — larva app

Muscle-driven larva locomotion simulator built on libuipc. This is the main app
of this fork; everything else supports it.

## Build & Run

One-time configure (from the repo root; needs CUDA >= 12.4 and the `CMAKE_TOOLCHAIN_FILE` vcpkg env var):

```bash
cmake -S . -B build --preset ci-release
```

Build and run (from the repo root):

```bash
cmake --build build --target larva -j 12 && build/Release/bin/larva <config>.json
```

- `--target larva` builds only this app and what it links. A bare
  `cmake --build build -j 12` also builds every test, example and backend.
- The config file name is resolved relative to this folder (`CONFIG_FOLDER`),
  so the binary can be launched from anywhere. Without an argument,
  `hunch.json` is used.
- Requires the CUDA backend (RTX-class GPU; `UIPC_WITH_CUDA_BACKEND=ON` is part
  of the `ci-release` preset); the `none` backend is not exercised by this app.
- Output goes to `output/examples/larva/main.cpp/` (not `build/`).

XMake parity: `xmake f -c && xmake build -j8 larva`, binary at
`build/linux/x86_64/release/larva`.

## Testing

There is no Catch2 test for this app — testing it means running the full
drop → hunch pipeline (see *Two-stage workflow* above) and checking/comparing
the exported outputs.

### Full pipeline run (from the repo root)

```bash
# 1. Drop: the larva falls from the original rest pose and settles on the ground
build/Release/bin/larva drop.json
# 2. Hand-off: drop's final pose becomes hunch's frame-0 geometry
cp output/examples/larva/main.cpp/larva_curr.msh apps/examples/larva/larva_curr.msh
# 3. Hunch: peristalsis driven by muscle_pattern.csv, starting from the settled pose
build/Release/bin/larva hunch.json
```

The hunch run is healthy when it exits 0 and writes per-frame
`*_vertices*.npy` plus `segment_lengths.csv` (both configs set
`segment_interfaces`) to the shared output folder. Watch stderr for
`logger::warn` about dropped csv muscles — expected only when the csv and mesh
muscle sets intentionally differ.

### `larva_curr.msh` hand-off

Only runs that start from the rest pose (no `curr_geometry` key, i.e. drop)
export `larva_curr.msh` at the end; hunch runs never do. This is deliberate:
a hunch export would hold the deformed end-of-peristalsis pose, and copying
that over the committed `apps/examples/larva/larva_curr.msh` would silently
give every later hunch a wrong frame-0 pose. Consequences:

- Copy the exported file back only after a **drop** run (as in the pipeline
  above); a hunch run produces nothing to copy.
- If `larva_curr.msh` was clobbered anyway, the committed copy is
  git-tracked: recover with `git checkout -- apps/examples/larva/larva_curr.msh`.

### A/B comparison

The sim is **not bit-deterministic between identical runs** (GPU
contact-solver atomics). Judge diffs against these measured noise floors
(max |diff| over 102 frames, `hunch.json`):

- raw interface mean-X columns: ~1.8e-05
- derived segment-length columns: ~1e-06 — these are the sensitive metrics

### Library changes

When the change touches engine/constitution/backend code rather than this app,
also run the repo test suite, e.g.
`ctest --test-dir build --build-config Release --label-regex fast --output-on-failure --timeout 900`
(see the `cmake-workflow` skill for details).

## Two-stage workflow

1. **Drop** (`drop.json`): the rest-pose larva falls onto the ground plane
   (`ground_height`) under gravity until contact. Exports `larva_curr.msh` to
   the output folder.
2. Copy the exported mesh into this folder as the hunch config's
   `curr_geometry` (`larva_curr.msh`). Only its vertex positions are used;
   topology and attributes come from the rest mesh.
3. **Hunch** (`hunch.json`): muscle activations from `muscle_pattern.csv`
   drive peristalsis starting from the settled pose. Frame count comes from
   the CSV time span resampled at `dt` (e.g. 1 s at dt=0.01 → 101 frames),
   not from `nb_frames`.
4. **Pack** (see below) for the viewer.

The output folder is shared and overwritten by every run — pack or move the
frames between runs (`pack.py` requires contiguous frame numbering starting
at 0, so stale files from a longer previous run will break it).

## Viewer export (`pack.py`)

The output is packaged for the larva-viewer at `~/code/larva-viewer`:

```bash
# muscle_pattern.csv must sit in the input folder so the viewer can show the
# activation curve; run this from apps/examples/larva after a hunch run
cp muscle_pattern.csv ../../../output/examples/larva/main.cpp/
python3 pack.py --data ~/code/larva-viewer/data \
    --input ../../../output/examples/larva/main.cpp --name <sim> --label "<label>"
```

- Produces `data/<sim>/{larva,muscle}_vertices.lvf.gz` and updates
  `data/manifest.json`. Simulations follow the `hunch`, `hunch2`, ... naming.
- The viewer has **one shared set of static assets** at its data root:
  `larva_surface.obj`, `muscle_surface.obj`, `muscle_groups.json`. Each run
  writes `larva_surface0.obj`, `muscle_surface0.obj` and `muscle_groups.json`
  to the output folder; after a mesh change, copy them over the ones at the
  data root (renaming the `0` suffix away). Old simulations whose surface
  vertex count differs will no longer render correctly — that's a viewer
  design limitation.
- Failure signature: the viewer throws e.g. `muscle: packed file has 42839
  vertices, mesh has 40288`. The first number is the packed sim's vertex
  count, the second the static obj's at the data root (`packed.js` compares
  the two) — they differ only when the root assets are from an older mesh,
  and the fix is the asset copy above. If the counts on disk already match,
  the browser is serving a cached obj: hard-refresh.

## Files

| File | Role |
|------|------|
| `main.cpp` | Simulation setup driven by a json config |
| `read_activation_csv.h` | CSV parsing, resampling to `dt` |
| `mesh_export.h` | `MeshExporter` (npy vertex export, msh save, muscle-group json) |
| `segment_tracker.h` | `SegmentTracker` (segment_lengths.csv export from interface vertex ids) |
| `fem_energy.csv` (output) | Per-frame energy bookkeeping via `FEMSystemFeature`: SNH and Muscle elastic energies (both divided by the exports' dt^2 scaling), kinetic, gravitational potential (datum at the ground plane) and contact energy. `elastic_energy` is relative to the frame-0 pose — for a drop run the constitutions' rest offset (SNK1's nonzero rest energy plus the Muscle passive rest energy) is subtracted, so it starts at 0. Read from Python with `numpy.genfromtxt(..., names=True)`. |
| `pack.py` | Not in this folder — it lives at `~/code/larva-viewer/pack.py`; it packs the exported `*_vertices*.npy` into `.lvf.gz` for the viewer |
| `larva.msh` | Rest mesh (tetrahedral, with muscle data) |
| `larva_curr.msh` | Frame-0 pose for the hunch (from a drop run) |
| `segment_interfaces.json` | Tracked interface vertex indices (band of tet-mesh vertices near each segment boundary plane, from the pipeline) |
| `muscle_pattern.csv` | Per-muscle activation time series |

## Mesh format

Binary Gmsh 2.2 as written by `igl::writeMSH` (read via `SimplicialComplexIO`),
with two per-tetrahedron `$ElementData` fields:

- `muscle_id` (scalar): `> 0` = muscle tet with that id, `-1` = passive.
- `direction` (3-vector): muscle contraction direction used by the `Muscle`
  constitution.

**Scale:** meshes come out of the segmentation pipeline at extent ~1.0. The
pipeline can now emit the mesh directly at simulation scale
(`generate_tet_mesh.py --length 0.005`); the committed `larva.msh` is at
that scale. Rescaling by hand is only needed for meshes produced without
`--length`.

**Segment interfaces:** `generate_tet_mesh.py` writes `segment_interfaces.json`
(interface vertex indices + which boundary pairs define segment lengths
A1..A3). The mesh is **not** cut: the tracked vertices are the tet-mesh
vertices within a small band (`TRACKING_BAND_TOL = 0.005` unscaled) of each
segment boundary plane (T3-A1, A1-A2, A2-A3, A3-A4) — material points the
`SegmentTracker` (`segment_tracker.h`) averages per frame. Plane-cutting the
mesh was tried and reverted: the cut decomposition creates ~1e-9-edge sliver
tets that destabilize the solve (see *Post-merge vertex pops* below). When the
config has a `segment_interfaces` key, `SegmentTracker` appends the mean X of
each interface and the A1/A2/A3 lengths (and their change vs frame 0) to
`segment_lengths.csv` every frame. Rest-frame lengths carry a small band bias
(≤ 4e-7, measured) that cancels in the `dlength_*` columns.

## Muscles & activations

- `muscle_pattern.csv`: first column is time in seconds, remaining headers are
  muscle ids. It is linearly resampled to the simulation `dt`.
- The muscle set of the mesh and the csv columns need not match exactly;
  `MuscleControllerFeature::read_activations()` reconciles them itself:
  - mesh muscles with no csv column → **zero activation** (silently, that is
    the documented semantic);
  - csv columns for muscles absent from the mesh → **dropped**, with a
    `logger::warn` listing the ids.
- `muscle_pattern.csv` is padded to exactly match the mesh: it has one column
  per muscle id of `larva.msh` (564; columns for mesh muscles without data
  are zero-filled, columns for muscles absent from the mesh are removed).

## Code style (app-specific)

- No single-use aggregate structs for return values: prefer plain types like
  `std::tuple<std::vector<int>, std::vector<int>>` with structured bindings.

## Config keys (json)

| Key | Meaning |
|-----|---------|
| `rest_geometry` | Rest mesh file (required) |
| `curr_geometry` | Frame-0 positions; omit to start at rest pose |
| `dt` | Time step (required) |
| `gravity` | bool; along −y |
| `ground`, `ground_height` | Add infinite half-plane floor at that height |
| `d_hat` | IPC contact activation distance |
| `kappa_eval_scale` | Sets `contact/adaptive/kappa_eval_scale`. Upstream now clamps contact kappa into a scene-adaptive corridor (Stiff-GIPC); without this key the 1.0_GPa default is clamped to ~1.6e4, which is ~4x slower (PCG needs ~2.7x more iterations) and softer. `6.29e-12` moves the corridor to [1e7, 1e9], restoring the pre-merge stiffness and speed. Both `drop.json` and `hunch.json` must use the same value (the settled pose depends on it). |
| `youngs_modulus`, `poisson_ratio` | StableNeoHookean parameters |
| `muscle_passive` | Passive muscle stiffness |
| `muscle_active` | Active modulus scale (activations are multiplied by it) |
| `muscle_contraction_pattern` | Activation csv; presence + `muscle_active > 0` enables muscle control |
| `segment_interfaces` | Interface vertex-index json; enables the `segment_lengths.csv` export |
| `nb_frames` | Frame count when muscle control is disabled |
| `engine_config` | Optional json object deep-merged into the scene config (RFC 7396) — diagnostics/A-B hook for engine keys not surfaced above, e.g. `{"newton": {"semi_implicit": {"enable": 0}}}`. Inert when absent. |

## Known pitfalls

- **Post-merge vertex pops (2026-11-15 diagnosis; 2026-11-15 resolved — mesh, not engine).** Runs with the merged engine
  (v0.0.29, Stiff-GIPC alignment) starting from a pose settled by the same
  engine show single-vertex teleport spikes (~1-8e-6, up to ~600x the frame's
  median motion), concentrated on one flank, in frames ~30-90 of the hunch.
  A/B results: unrelated to CUDA-graph PCG replay, broad-phase filter choice,
  or silent line-search failures (strict mode clean). Setting
  `engine_config: {"newton": {"semi_implicit": {"enable": 0}, "min_iter": 1}}`
  removes ~85% of them but roughly doubles per-frame runtime (median
  548 -> 1206 ms) and exposes a slow lateral COM drift instead. Lateral
  (z) symmetry breaking is **not** a merge regression: the pre-merge engine
  drifts COM z by up to ~5e-4 (and the pre-merge-settled `larva_curr.msh` was
  itself settled ~4.5e-4 off-axis), while post-merge hunch stays within 3.5e-5.
  **Resolution:** the pops were caused by the rest mesh, not the engine —
  `larva.msh` (commit 06f1d817) had been sliced along the segment boundary
  planes by the pipeline, and the cut decomposition manufactures ~8800
  sliver tets (edges down to ~6e-10 vs `d_hat = 2e-05`). A/B on the same
  fTetWild output: the uncut twin shows 0 jump frames (ratio max ~28) and
  settles symmetrically. The pipeline now tracks interfaces via vertex
  bands instead of cutting (see *Segment interfaces* above), and the app
  mesh was swapped to the uncut one.
  Analysis tooling: `output/larva_debug/analyze_run.py` (jump/symmetry/COM
  metrics from a run's npy frames); the A/B evidence lives in
  `output/larva_debug/ab_*` (cut vs nocut vs band variants).

- A mesh starting below `ground_height` aborts init with
  "Geometry is too close (distance <= 0) to HalfPlane" (check the mesh scale).
- `main.cpp` copies only vertices from `curr_geometry` onto the rest mesh, so
  the curr mesh's own attributes (`direction` etc.) are never used.
- Muscle ids ≤ 0 are treated as passive everywhere (`read_muscle_groups_from`,
  `MeshExporter`).
