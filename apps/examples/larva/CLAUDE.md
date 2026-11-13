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

1. **Drop** (`drop.json` / `drop_new.json`): the rest-pose larva falls onto the
   ground plane (`ground_height`) under gravity until contact. Exports
   `larva_curr.msh` to the output folder.
2. Copy the exported mesh into this folder as the hunch config's
   `curr_geometry` (e.g. `larva_curr.msh` / `larva_curr_new.msh`).
   Only its vertex positions are used; topology and attributes come from the
   rest mesh.
3. **Hunch** (`hunch.json` / `hunch_new.json`): muscle activations from
   `muscle_pattern.csv` drive peristalsis starting from the settled pose.
   Frame count comes from the CSV time span resampled at `dt` (e.g. 1 s at
   dt=0.01 → 101 frames), not from `nb_frames`.

The output folder is shared and overwritten by every run — pack or move the
frames between runs (`pack.py` requires contiguous frame numbering starting
at 0, so stale files from a longer previous run will break it).

## Files

| File | Role |
|------|------|
| `main.cpp` | Simulation setup driven by a json config |
| `read_activation_csv.h` | CSV parsing, resampling to `dt`, mesh↔csv muscle reconciliation |
| `mesh_export.h` | `MeshExporter` (npy vertex export, msh save, muscle-group json) |
| `pack.py` | Packs `larva_vertices*.npy`/`muscle_vertices*.npy` into `.lvf` for the viewer (`output/view.py`) |
| `larva.msh`, `larva_new.msh` | Rest meshes (tetrahedral, with muscle data) |
| `muscle_pattern.csv` | Per-muscle activation time series |

## Mesh format

Binary Gmsh 2.2 as written by `igl::writeMSH` (read via `SimplicialComplexIO`),
with two per-tetrahedron `$ElementData` fields:

- `muscle_id` (scalar): `> 0` = muscle tet with that id, `-1` = passive.
- `direction` (3-vector): muscle contraction direction used by the `Muscle`
  constitution.

**Scale:** meshes come out of the segmentation pipeline at extent ~1.0.
Before simulation they must be rescaled by **0.005** (about the origin) to
match the tuned parameters (`d_hat = 2e-5`, `ground_height = -0.001`); see
commit `4c08a964` "rescale mesh". `larva.msh` is already rescaled;
`larva_new.msh` was rescaled the same way.

## Muscles & activations

- `muscle_pattern.csv`: first column is time in seconds, remaining headers are
  muscle ids. It is linearly resampled to the simulation `dt`.
- The muscle set of the mesh and the csv columns need not match exactly.
  `match_activations_to_muscles()` (called from `main.cpp` after resampling)
  reconciles them and prints what it did:
  - mesh muscles with no csv column → **zero activation**;
  - csv columns for muscles absent from the mesh → **dropped** (otherwise the
    muscle controller would index out of bounds).
- Reference counts: `larva.msh` has 452 muscle ids, exactly matching the csv.
  `larva_new.msh` has 564 ids: 114 are csv-less (zero-activated) and ids
  223/273 are absent from the mesh (dropped).

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
| `nb_frames` | Frame count when muscle control is disabled |

## Known pitfalls

- A mesh starting below `ground_height` aborts init with
  "Geometry is too close (distance <= 0) to HalfPlane" (check the mesh scale).
- `main.cpp` copies only vertices from `curr_geometry` onto the rest mesh, so
  the curr mesh's own attributes (`direction` etc.) are never used.
- Muscle ids ≤ 0 are treated as passive everywhere (`read_muscle_groups_from`,
  `MeshExporter`).
