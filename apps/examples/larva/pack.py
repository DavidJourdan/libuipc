#!/usr/bin/env python3
"""Pack per-frame .npy vertex files into .lvf(.gz) and register them in the manifest.

Import a fresh simulation run from elsewhere:

    python pack.py --input ../larva-simulator/output --name wave_fast \
                   --label "fast wave, 2 Hz"

Re-pack everything already sitting in data/:

    python pack.py --rebuild
"""

import argparse
import datetime
import gzip
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np

MAGIC = b"LVF2"
ENCODING_RAW_FLOAT32 = 0
ENCODING_DELTA_INT16 = 1
HEADER_SIZE = 32

LAYER_STEMS = ("larva_vertices", "muscle_vertices")
PATTERN_NAMES = ("muscle_pattern.csv", "muscle_pattern.json")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load_frame_series(folder: Path, stem: str) -> np.ndarray:
    pattern = re.compile(rf"^{re.escape(stem)}(\d+)\.npy$")
    matches = []

    for path in folder.iterdir():
        found = pattern.match(path.name)
        if found is not None:
            matches.append((int(found.group(1)), path))

    if not matches:
        raise FileNotFoundError(f"no {stem}*.npy in {folder}")

    matches.sort(key=lambda item: item[0])

    indices = [index for index, _ in matches]
    if indices != list(range(len(indices))):
        missing = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise ValueError(
            f"{stem}: frame numbering is not contiguous from 0 "
            f"(found {len(indices)} files, highest index {indices[-1]}, "
            f"missing {missing[:10]})"
        )

    frames = []
    shape = None

    for index, path in matches:
        frame = np.load(path).astype(np.float32, copy=False)

        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(
                f"{stem}: frame {index} has shape {frame.shape}, expected {shape}"
            )

        frames.append(frame)

    return np.stack(frames).reshape(len(frames), -1)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def decode_delta_int16(first_frame, deltas, scale):
    frames = np.empty((len(deltas) + 1, first_frame.size), dtype=np.float32)
    frames[0] = first_frame

    for index in range(len(deltas)):
        frames[index + 1] = frames[index] + deltas[index].astype(np.float32) * scale

    return frames


def encode_delta_int16(frames: np.ndarray, headroom: float = 1.08):
    frame_count, component_count = frames.shape

    raw_deltas = np.diff(frames, axis=0)

    if raw_deltas.size == 0:
        largest = 0.0
    else:
        largest = float(np.abs(raw_deltas).max())

    if largest == 0.0:
        scale = 1.0
    else:
        scale = largest * headroom / 32767.0

    deltas = np.zeros((frame_count - 1, component_count), dtype=np.int16)
    reconstructed = frames[0].astype(np.float32).copy()
    clipped_count = 0

    # Encode against the reconstruction rather than the original, so the
    # residual feeds back and the error stays bounded by scale/2 per frame
    # instead of accumulating along the series.
    for index in range(1, frame_count):
        quantised = np.rint((frames[index] - reconstructed) / scale)

        clipped_count += int(np.count_nonzero(np.abs(quantised) > 32767))
        np.clip(quantised, -32768, 32767, out=quantised)

        deltas[index - 1] = quantised.astype(np.int16)
        reconstructed += deltas[index - 1].astype(np.float32) * scale

    if clipped_count > 0:
        raise ValueError(
            f"{clipped_count} deltas clipped at int16 range — increase headroom "
            f"or pass --raw for this dataset"
        )

    check = decode_delta_int16(frames[0], deltas, scale)
    worst = float(np.abs(frames - check).max())

    return deltas, scale, worst


def write_packed(path: Path, frames: np.ndarray, encoding: int, compress: bool) -> Path:
    frame_count, component_count = frames.shape
    vertex_count = component_count // 3

    if encoding == ENCODING_DELTA_INT16:
        deltas, scale, worst = encode_delta_int16(frames)
        payload = frames[0].tobytes() + deltas.tobytes()
        extent = float(np.abs(frames).max())
        relative = 0.0
        if extent > 0:
            relative = worst / extent
        print(f"    quantisation error {worst:.3e} "
              f"({relative * 100:.4f}% of extent, scale {scale:.3e})")
    else:
        scale = 0.0
        payload = np.ascontiguousarray(frames, dtype=np.float32).tobytes()

    header = struct.pack(
        "<4sIIIIfII",
        MAGIC,
        1,
        frame_count,
        vertex_count,
        encoding,
        scale,
        0,
        0,
    )
    assert len(header) == HEADER_SIZE

    blob = header + payload

    if compress:
        target = path.with_suffix(path.suffix + ".gz")
        with gzip.open(target, "wb", compresslevel=6) as handle:
            handle.write(blob)
    else:
        target = path
        target.write_bytes(blob)

    # Remove the other variant so a stale file can never shadow this one.
    if compress:
        stale = path
    else:
        stale = path.with_suffix(path.suffix + ".gz")
    if stale.exists():
        stale.unlink()
        print(f"    removed stale {stale.name}")

    original_bytes = frame_count * component_count * 4
    packed_bytes = target.stat().st_size
    print(f"    {target.name}: {packed_bytes / 1e6:.1f} MB "
          f"({original_bytes / packed_bytes:.1f}x smaller)")

    return target


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())

    return {
        "version": 1,
        "staticAssets": {
            "larvaSurface": "larva_surface.obj",
            "muscleSurface": "muscle_surface.obj",
            "muscleGroups": "muscle_groups.json",
        },
        "simulations": [],
    }


def upsert_simulation(manifest: dict, entry: dict) -> None:
    simulations = manifest.setdefault("simulations", [])

    for index, existing in enumerate(simulations):
        if existing["name"] == entry["name"]:
            merged = dict(existing)
            merged.update(entry)
            simulations[index] = merged
            print(f"  updated manifest entry '{entry['name']}'")
            return

    simulations.append(entry)
    print(f"  added manifest entry '{entry['name']}'")


def pattern_duration(path: Path):
    if path.suffix == ".csv":
        times = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, ndmin=1)
        return round(float(times[-1]), 6)

    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "time" in payload:
        return round(float(payload["time"][-1]), 6)

    return None


def auto_name(data_root: Path, source: Path) -> str:
    candidate = source.resolve().name

    # "output", "out", "results" tell us nothing, so fall back to a timestamp.
    if candidate.lower() in {"output", "out", "results", "data", "npy", "frames"}:
        candidate = source.resolve().parent.name

    if candidate.lower() in {"", "/", "home", "tmp"}:
        candidate = "run"

    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("_")

    if not (data_root / candidate).exists():
        return candidate

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{candidate}_{stamp}"


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------

def pack_folder(folder: Path, encoding: int, compress: bool, remove_npy: bool) -> int:
    frame_counts = {}

    for stem in LAYER_STEMS:
        frames = load_frame_series(folder, stem)
        frame_counts[stem] = frames.shape[0]
        print(f"  {stem}: {frames.shape[0]} frames x {frames.shape[1] // 3} vertices")
        write_packed(folder / f"{stem}.lvf", frames, encoding, compress)

        if remove_npy:
            for path in folder.glob(f"{stem}*.npy"):
                path.unlink()

    counts = set(frame_counts.values())
    if len(counts) > 1:
        print(f"  ** WARNING: layers disagree on frame count: {frame_counts}")

    return max(counts)


def import_simulation(arguments) -> None:
    source = arguments.input.resolve()
    data_root = arguments.data

    if not source.is_dir():
        raise FileNotFoundError(f"input folder does not exist: {source}")

    name = arguments.name
    if name is None:
        name = auto_name(data_root, source)

    destination = data_root / name

    if destination.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"{destination} already exists — pass --overwrite to replace it"
        )

    destination.mkdir(parents=True, exist_ok=True)
    print(f"importing {source} -> {destination}")

    if arguments.raw:
        encoding = ENCODING_RAW_FLOAT32
    else:
        encoding = ENCODING_DELTA_INT16

    frame_count = None

    for stem in LAYER_STEMS:
        frames = load_frame_series(source, stem)
        print(f"  {stem}: {frames.shape[0]} frames x {frames.shape[1] // 3} vertices")

        if frame_count is not None and frames.shape[0] != frame_count:
            print(f"  ** WARNING: {stem} has {frames.shape[0]} frames, "
                  f"previous layer had {frame_count}")
        frame_count = frames.shape[0]

        write_packed(destination / f"{stem}.lvf", frames, encoding,
                     not arguments.no_gzip)

    pattern_source = None
    for candidate in PATTERN_NAMES:
        if (source / candidate).exists():
            pattern_source = source / candidate
            break

    duration = None

    if pattern_source is None:
        print(f"  ** WARNING: no {' or '.join(PATTERN_NAMES)} in {source}")
    else:
        shutil.copy2(pattern_source, destination / pattern_source.name)
        print(f"    copied {pattern_source.name}")
        duration = pattern_duration(pattern_source)

    if arguments.remove_source:
        for stem in LAYER_STEMS:
            for path in source.glob(f"{stem}*.npy"):
                path.unlink()
        print(f"  removed source .npy files from {source}")

    entry = {"name": name, "frameCount": frame_count}

    if arguments.label is not None:
        entry["label"] = arguments.label
    else:
        entry["label"] = name

    if duration is not None:
        entry["duration"] = duration

    if arguments.notes is not None:
        entry["notes"] = arguments.notes

    entry["importedFrom"] = str(source)
    entry["importedAt"] = datetime.datetime.now().isoformat(timespec="seconds")

    manifest_path = data_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    upsert_simulation(manifest, entry)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {manifest_path}")


def rebuild_all(arguments) -> None:
    data_root = arguments.data

    if arguments.raw:
        encoding = ENCODING_RAW_FLOAT32
    else:
        encoding = ENCODING_DELTA_INT16

    manifest_path = data_root / "manifest.json"
    manifest = load_manifest(manifest_path)

    folders = sorted(path for path in data_root.iterdir() if path.is_dir())

    for folder in folders:
        has_npy = any(folder.glob("*_vertices*.npy"))
        if not has_npy:
            print(f"{folder.name}: no .npy frames, skipping")
            continue

        print(f"{folder.name}:")
        frame_count = pack_folder(
            folder, encoding, not arguments.no_gzip, arguments.remove_source
        )

        entry = {"name": folder.name, "frameCount": frame_count}

        for candidate in PATTERN_NAMES:
            if (folder / candidate).exists():
                duration = pattern_duration(folder / candidate)
                if duration is not None:
                    entry["duration"] = duration
                break

        upsert_simulation(manifest, entry)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {manifest_path} with {len(manifest['simulations'])} simulations")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack simulation vertex frames for the larva viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="viewer data root (default: data)")
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="external folder holding *_vertices*.npy to import")
    parser.add_argument("--name", "-n", default=None,
                        help="destination folder name under data/ "
                             "(default: derived from the input folder)")
    parser.add_argument("--label", "-l", default=None,
                        help="human-readable label shown in the dropdown")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--rebuild", action="store_true",
                        help="re-pack every simulation folder already in data/")
    parser.add_argument("--raw", action="store_true",
                        help="store float32 instead of quantised int16 deltas")
    parser.add_argument("--no-gzip", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing destination folder")
    parser.add_argument("--remove-source", action="store_true",
                        help="delete the .npy files once packed")

    arguments = parser.parse_args()

    if arguments.input is not None:
        import_simulation(arguments)
    elif arguments.rebuild:
        rebuild_all(arguments)
    else:
        parser.error("pass --input to import a run, or --rebuild to re-pack data/")


if __name__ == "__main__":
    main()