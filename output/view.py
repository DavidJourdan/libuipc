import glob
import os
import re
import sys
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# ── Configuration ──────────────────────────────────────────────────────────────
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."
DEFAULT_FPS = 25.0
# ──────────────────────────────────────────────────────────────────────────────


# ── Minimal OBJ loader ─────────────────────────────────────────────────────────
def load_obj(path: str):
    """
    Returns (vertices [N,3], faces [M,3]).
    Handles v/vt/vn face entries and fan-triangulates polygons.
    Replace this with meshio.read() / trimesh.load() if you need richer support.
    """
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith("#"):
                continue
            if tok[0] == "v":
                verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] == "f":
                ids = [
                    int(t.split("/")[0]) - 1 for t in tok[1:]
                ]  # 1-indexed → 0-indexed
                for i in range(1, len(ids) - 1):  # fan triangulation
                    faces.append((ids[0], ids[i], ids[i + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)

def load_msh(path: str):
    mesh = meshio.read(path)

    V = mesh.points
    F = mesh.cells[0].data
    activations = mesh.cell_data["active_modulus"]

    return V, F, activations


# ── File discovery ─────────────────────────────────────────────────────────────
def discover_files(folder: str, pattern: str):
    """Return files sorted by the integer embedded in their name."""
    paths = glob.glob(os.path.join(folder, pattern))

    def _key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(paths, key=_key)


surface_files = discover_files(FOLDER, "larva_vertices*.npy")
muscle_files = discover_files(FOLDER, "muscle_vertices*.npy")

N = len(surface_files)
print(f"Found {N} frames in '{FOLDER}'")

if len(muscle_files) != N:
    print(f"Error: surface doesn't match muscles ({len(muscle_files)} != {N}).")


# ── Polyscope init ─────────────────────────────────────────────────────────────
ps.init()
ps.set_program_name("Simulation Surface Viewer")
ps.set_give_focus_on_show(True)
ps.set_ground_plane_mode('shadow_only')
ps.set_ground_plane_height(-0.001)
ps.set_transparency_render_passes(2)

# State (plain dict so the nested callback can mutate it freely)
state = {
    "frame": 0,
    "playing": False,
    "fps": DEFAULT_FPS,
    "last_t": 0.0,
}

verts, faces = load_obj(os.path.join(FOLDER, "larva_surface.obj"))
surface_mesh = ps.register_surface_mesh(
    "surface", verts, faces, smooth_shade=False, edge_width=1
)
surface_mesh.set_transparency(0.5)
surface_mesh.set_edge_width(0.0)
verts, faces = load_obj(os.path.join(FOLDER, "muscle_surface.obj"))
muscle_mesh = ps.register_surface_mesh(
    "muscle", verts, faces, smooth_shade=False, edge_width=1
)

# surface_verts = []
# muscle_verts = []
# for idx in range(N):
#     verts, _ = load_obj(surface_files[idx])
#     surface_verts.append(verts)

#     verts, _ = load_obj(muscle_files[idx])
#     muscle_verts.append(verts)




def show_frame(idx: int):
    """Load OBJ at index idx and push it to polyscope."""
    # surface_mesh.update_vertex_positions(surface_verts[idx])
    # muscle_mesh.update_vertex_positions(muscle_verts[idx])
    surface_mesh.update_vertex_positions(np.load(surface_files[idx]))
    muscle_mesh.update_vertex_positions(np.load(muscle_files[idx]))

show_frame(0)

# ── UI callback ────────────────────────────────────────────────────────────────
def ui_callback():
    s = state

    # ── Transport buttons ────────────────────────────────────────────────────
    if s["playing"]:
        if psim.Button("Pause"):
            s["playing"] = False
    else:
        if psim.Button("Play "):
            s["playing"] = True
            s["last_t"] = time.time()

    psim.SameLine()
    if psim.Button("Prev"):
        s["playing"] = False
        s["frame"] = max(0, s["frame"] - 1)
        show_frame(s["frame"])

    psim.SameLine()
    if psim.Button("Next"):
        s["playing"] = False
        s["frame"] = min(N - 1, s["frame"] + 1)
        show_frame(s["frame"])

    psim.SameLine()
    if psim.Button("Reset"):
        s["playing"] = False
        s["frame"] = 0
        show_frame(s["frame"])

    # ── Frame scrubber ───────────────────────────────────────────────────────
    changed, val = psim.SliderInt("Frame", s["frame"], 0, N - 1)
    if changed:
        s["playing"] = False
        s["frame"] = val
        show_frame(s["frame"])

    # ── Playback speed ───────────────────────────────────────────────────────
    _, s["fps"] = psim.SliderFloat("FPS", s["fps"], 1.0, 60.0)

    # ── Info line ────────────────────────────────────────────────────────────
    psim.TextUnformatted(
        f"Frame {s['frame'] + 1} / {N}   —   {os.path.basename(surface_files[s['frame']])}"
    )

    # ── Auto-advance when playing ────────────────────────────────────────────
    if s["playing"]:
        now = time.time()
        if now - s["last_t"] >= 1.0 / s["fps"]:
            s["frame"] = (s["frame"] + 1) % N  # loops back to 0
            show_frame(s["frame"])
            s["last_t"] = now


ps.set_user_callback(ui_callback)
ps.show()
