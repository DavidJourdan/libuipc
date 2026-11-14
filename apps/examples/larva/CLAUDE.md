# CLAUDE.md — larva app

Muscle-driven larva locomotion simulator built on libuipc. This is the main app
of this fork; everything else supports it.

## Build & Run

```bash
cmake --build build -j 12 && build/Release/bin/larva <config>.json
```

- The config file name is resolved relative to this folder (`CONFIG_FOLDER`).
  Without an argument, `hunch.json` is used.
- Requires the CUDA backend (RTX-class GPU); the `none` backend is not
  exercised by this app.
- Output goes to `output/examples/larva/main.cpp/` (not `build/`).

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

## Files

| File | Role |
|------|------|
| `main.cpp` | Simulation setup driven by a json config |
| `read_activation_csv.h` | CSV parsing, resampling to `dt` |
| `mesh_export.h` | `MeshExporter` (npy vertex export, msh save, muscle-group json) |
| `segment_tracker.h` | `SegmentTracker` (segment_lengths.csv export from interface vertex ids) |
| `pack.py` | Not in this folder — it lives at `~/code/larva-viewer/pack.py`; it packs the exported `*_vertices*.npy` into `.lvf.gz` for the viewer |
| `larva.msh` | Rest mesh (tetrahedral, with muscle data) |
| `larva_curr.msh` | Frame-0 pose for the hunch (from a drop run) |
| `segment_interfaces.json` | Interface vertex indices on the segment boundary planes (from the pipeline) |
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

**Segment interfaces:** `generate_tet_mesh.py` also cuts the mesh along the
segment boundary planes (T3-A1, A1-A2, A2-A3, A3-A4) so vertices sit exactly
on them, and writes `segment_interfaces.json` (interface vertex indices +
which boundary pairs define segment lengths A1..A3). When the config has a
`segment_interfaces` key, `SegmentTracker` (`segment_tracker.h`) appends the
mean X of each interface and the A1/A2/A3 lengths (and their change vs
frame 0) to `segment_lengths.csv` every frame.

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
| `youngs_modulus`, `poisson_ratio` | StableNeoHookean parameters |
| `muscle_passive` | Passive muscle stiffness |
| `muscle_active` | Active modulus scale (activations are multiplied by it) |
| `muscle_contraction_pattern` | Activation csv; presence + `muscle_active > 0` enables muscle control |
| `segment_interfaces` | Interface vertex-index json; enables the `segment_lengths.csv` export |
| `nb_frames` | Frame count when muscle control is disabled |

## Known pitfalls

- A mesh starting below `ground_height` aborts init with
  "Geometry is too close (distance <= 0) to HalfPlane" (check the mesh scale).
- `main.cpp` copies only vertices from `curr_geometry` onto the rest mesh, so
  the curr mesh's own attributes (`direction` etc.) are never used.
- Muscle ids ≤ 0 are treated as passive everywhere (`read_muscle_groups_from`,
  `MeshExporter`).
